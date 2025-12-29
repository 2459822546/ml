///|
/// 张量 (Tensor)
/// 多维数值数组，支持任意维度的张量运算
pub struct Tensor {
  data : Array[Double] // 扁平化存储
  shape : Array[Int] // 各维度大小
  strides : Array[Int] // 步长
  size : Int // 总元素数
}

///|
/// 构造函数
pub fn Tensor::new(data : Array[Double], shape : Array[Int]) -> Tensor {
  // 计算总大小
  let mut total_size = 1
  for i = 0; i < shape.length(); i = i + 1 {
    if shape[i] <= 0 {
      abort("张量维度必须大于0")
    }
    total_size = total_size * shape[i]
  }
  if data.length() != total_size {
    abort("数据长度与形状不匹配")
  }

  // 计算步长 (strides)
  let strides = Array::make(shape.length(), 0)
  let mut stride = 1
  for i = shape.length() - 1; i >= 0; i = i - 1 {
    strides[i] = stride
    stride = stride * shape[i]
  }
  { data, shape, strides, size: total_size }
}

///|
pub fn Tensor::zeros(shape : Array[Int]) -> Tensor {
  let mut size = 1
  for i = 0; i < shape.length(); i = i + 1 {
    size = size * shape[i]
  }
  Tensor::new(Array::make(size, 0.0), shape)
}

///|
pub fn Tensor::ones(shape : Array[Int]) -> Tensor {
  let mut size = 1
  for i = 0; i < shape.length(); i = i + 1 {
    size = size * shape[i]
  }
  Tensor::new(Array::make(size, 1.0), shape)
}

///|
pub fn Tensor::fill(shape : Array[Int], value : Double) -> Tensor {
  let mut size = 1
  for i = 0; i < shape.length(); i = i + 1 {
    size = size * shape[i]
  }
  Tensor::new(Array::make(size, value), shape)
}

///|
pub fn Tensor::from_vector(vec : Vector) -> Tensor {
  Tensor::new(vec.to_array(), [vec.size()])
}

///|
pub fn Tensor::from_matrix(mat : Matrix) -> Tensor {
  let (rows, cols) = mat.shape()
  let data = Array::make(rows * cols, 0.0)
  let mut idx = 0
  for i = 0; i < rows; i = i + 1 {
    for j = 0; j < cols; j = j + 1 {
      data[idx] = mat.get(i, j)
      idx = idx + 1
    }
  }
  Tensor::new(data, [rows, cols])
}

///|
/// 基本属性
pub fn Tensor::get_shape(self : Tensor) -> Array[Int] {
  self.shape
}

///|
pub fn Tensor::ndim(self : Tensor) -> Int {
  self.shape.length()
}

///|
pub fn Tensor::total_size(self : Tensor) -> Int {
  self.size
}

///|
/// 计算多维索引对应的扁平化索引
fn Tensor::compute_flat_index(self : Tensor, indices : Array[Int]) -> Int {
  if indices.length() != self.shape.length() {
    abort("索引维度不匹配")
  }
  let mut flat_idx = 0
  for i = 0; i < indices.length(); i = i + 1 {
    if indices[i] < 0 || indices[i] >= self.shape[i] {
      abort("索引越界")
    }
    flat_idx = flat_idx + indices[i] * self.strides[i]
  }
  flat_idx
}

///|
pub fn Tensor::get(self : Tensor, indices : Array[Int]) -> Double {
  let idx = self.compute_flat_index(indices)
  self.data[idx]
}

///|
pub fn Tensor::set(self : Tensor, indices : Array[Int], value : Double) -> Unit {
  let idx = self.compute_flat_index(indices)
  self.data[idx] = value
}

///| 张量运算

///|
/// 张量加法
pub fn Tensor::op_add(self : Tensor, other : Tensor) -> Tensor {
  let out_shape = broadcast_shape(self.shape, other.shape)
  let out_strides = compute_strides_for_shape(out_shape)
  let mut out_size = 1
  for i = 0; i < out_shape.length(); i = i + 1 {
    out_size = out_size * out_shape[i]
  }
  let result_data = Array::make(out_size, 0.0)
  for i = 0; i < out_size; i = i + 1 {
    let out_indices = indices_from_flat(out_shape, out_strides, i)
    let idx_a = broadcast_index(
      out_indices,
      out_shape,
      self.shape,
      self.strides,
    )
    let idx_b = broadcast_index(
      out_indices,
      out_shape,
      other.shape,
      other.strides,
    )
    result_data[i] = self.data[idx_a] + other.data[idx_b]
  }
  Tensor::new(result_data, out_shape)
}

///|
/// 张量减法
pub fn Tensor::op_sub(self : Tensor, other : Tensor) -> Tensor {
  let out_shape = broadcast_shape(self.shape, other.shape)
  let out_strides = compute_strides_for_shape(out_shape)
  let mut out_size = 1
  for i = 0; i < out_shape.length(); i = i + 1 {
    out_size = out_size * out_shape[i]
  }
  let result_data = Array::make(out_size, 0.0)
  for i = 0; i < out_size; i = i + 1 {
    let out_indices = indices_from_flat(out_shape, out_strides, i)
    let idx_a = broadcast_index(
      out_indices,
      out_shape,
      self.shape,
      self.strides,
    )
    let idx_b = broadcast_index(
      out_indices,
      out_shape,
      other.shape,
      other.strides,
    )
    result_data[i] = self.data[idx_a] - other.data[idx_b]
  }
  Tensor::new(result_data, out_shape)
}

///|
/// 张量除法
pub fn Tensor::op_div(self : Tensor, other : Tensor) -> Tensor {
  let out_shape = broadcast_shape(self.shape, other.shape)
  let out_strides = compute_strides_for_shape(out_shape)
  let mut out_size = 1
  for i = 0; i < out_shape.length(); i = i + 1 {
    out_size = out_size * out_shape[i]
  }
  let result_data = Array::make(out_size, 0.0)
  for i = 0; i < out_size; i = i + 1 {
    let out_indices = indices_from_flat(out_shape, out_strides, i)
    let idx_a = broadcast_index(
      out_indices,
      out_shape,
      self.shape,
      self.strides,
    )
    let idx_b = broadcast_index(
      out_indices,
      out_shape,
      other.shape,
      other.strides,
    )
    result_data[i] = self.data[idx_a] / other.data[idx_b]
  }
  Tensor::new(result_data, out_shape)
}

///|
/// 标量乘法
pub fn Tensor::scalar_mul(self : Tensor, scalar : Double) -> Tensor {
  let result_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    result_data[i] = self.data[i] * scalar
  }
  Tensor::new(result_data, self.shape)
}

///|
/// 元素级乘法 (Hadamard product) - 用于 * 运算符
pub fn Tensor::op_mul(self : Tensor, other : Tensor) -> Tensor {
  let out_shape = broadcast_shape(self.shape, other.shape)
  let out_strides = compute_strides_for_shape(out_shape)
  let mut out_size = 1
  for i = 0; i < out_shape.length(); i = i + 1 {
    out_size = out_size * out_shape[i]
  }
  let result_data = Array::make(out_size, 0.0)
  for i = 0; i < out_size; i = i + 1 {
    let out_indices = indices_from_flat(out_shape, out_strides, i)
    let idx_a = broadcast_index(
      out_indices,
      out_shape,
      self.shape,
      self.strides,
    )
    let idx_b = broadcast_index(
      out_indices,
      out_shape,
      other.shape,
      other.strides,
    )
    result_data[i] = self.data[idx_a] * other.data[idx_b]
  }
  Tensor::new(result_data, out_shape)
}

///|
/// Hadamard 积的别名（保持API一致性）
pub fn Tensor::hadamard(self : Tensor, other : Tensor) -> Tensor {
  self.op_mul(other)
}

///|
/// 实现 Add trait 以支持 + 运算符
pub impl Add for Tensor with add(self, other) {
  self.op_add(other)
}

///|
/// 实现 Sub trait 以支持 - 运算符
pub impl Sub for Tensor with sub(self, other) {
  self.op_sub(other)
}

///|
/// 实现 Mul trait 以支持 * 运算符（元素级乘法）
pub impl Mul for Tensor with mul(self, other) {
  self.op_mul(other)
}

///|
/// 实现 Div trait 以支持 / 运算符（元素级除法）
pub impl Div for Tensor with div(self, other) {
  self.op_div(other)
}

///| 形状操作

///|
/// 重塑张量
pub fn Tensor::reshape(self : Tensor, new_shape : Array[Int]) -> Tensor {
  let mut new_size = 1
  let mut inferred_idx = -1
  for i = 0; i < new_shape.length(); i = i + 1 {
    let dim = new_shape[i]
    if dim == -1 {
      if inferred_idx != -1 {
        abort("只能推断一个维度")
      }
      inferred_idx = i
      continue
    }
    if dim <= 0 {
      abort("张量维度必须大于0")
    }
    new_size = new_size * dim
  }
  let final_shape = Array::make(new_shape.length(), 0)
  for i = 0; i < new_shape.length(); i = i + 1 {
    final_shape[i] = new_shape[i]
  }
  if inferred_idx != -1 {
    if new_size == 0 || self.size % new_size != 0 {
      abort("无法推断形状")
    }
    final_shape[inferred_idx] = self.size / new_size
    new_size = new_size * final_shape[inferred_idx]
  }
  if new_size != self.size {
    abort("新形状的总大小必须与原张量相同")
  }

  // 复制数据
  let new_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    new_data[i] = self.data[i]
  }
  Tensor::new(new_data, final_shape)
}

///|
/// 转置（支持高阶维度）
pub fn Tensor::transpose(self : Tensor, axes? : Array[Int] = []) -> Tensor {
  let ndim = self.ndim()
  if ndim == 0 {
    return self.copy()
  }
  let perm = if axes.length() == 0 {
    let p = Array::make(ndim, 0)
    for i = 0; i < ndim; i = i + 1 {
      p[i] = ndim - 1 - i
    }
    p
  } else {
    axes
  }
  if perm.length() != ndim {
    abort("转置轴维度不匹配")
  }
  let seen = Array::make(ndim, false)
  for i = 0; i < perm.length(); i = i + 1 {
    let ax = perm[i]
    if ax < 0 || ax >= ndim {
      abort("转置轴越界")
    }
    if seen[ax] {
      abort("转置轴重复")
    }
    seen[ax] = true
  }
  let new_shape = Array::make(ndim, 0)
  for i = 0; i < ndim; i = i + 1 {
    new_shape[i] = self.shape[perm[i]]
  }
  let new_strides = compute_strides_for_shape(new_shape)
  let new_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    let old_indices = indices_from_flat(self.shape, self.strides, i)
    let new_indices = Array::make(ndim, 0)
    for j = 0; j < ndim; j = j + 1 {
      new_indices[j] = old_indices[perm[j]]
    }
    let mut new_idx = 0
    for j = 0; j < ndim; j = j + 1 {
      new_idx = new_idx + new_indices[j] * new_strides[j]
    }
    new_data[new_idx] = self.data[i]
  }
  Tensor::new(new_data, new_shape)
}

///|
/// 沿指定轴切片 [start, end)
pub fn Tensor::slice(
  self : Tensor,
  axis : Int,
  start : Int,
  end? : Int = -1,
) -> Tensor {
  if axis < 0 || axis >= self.ndim() {
    abort("轴索引越界")
  }
  let axis_len = self.shape[axis]
  let mut stop = end
  if stop == -1 {
    stop = axis_len
  }
  if start < 0 || stop < start || stop > axis_len {
    abort("切片范围非法")
  }
  let new_shape = Array::make(self.shape.length(), 0)
  for i = 0; i < self.shape.length(); i = i + 1 {
    new_shape[i] = self.shape[i]
  }
  new_shape[axis] = stop - start
  let new_strides = compute_strides_for_shape(new_shape)
  let mut new_size = 1
  for i = 0; i < new_shape.length(); i = i + 1 {
    new_size = new_size * new_shape[i]
  }
  let new_data = Array::make(new_size, 0.0)
  for idx = 0; idx < new_size; idx = idx + 1 {
    let out_indices = indices_from_flat(new_shape, new_strides, idx)
    let mut in_idx = 0
    for d = 0; d < self.ndim(); d = d + 1 {
      let v = if d == axis { out_indices[d] + start } else { out_indices[d] }
      in_idx = in_idx + v * self.strides[d]
    }
    new_data[idx] = self.data[in_idx]
  }
  Tensor::new(new_data, new_shape)
}

///|
/// 沿指定轴拼接
pub fn Tensor::concat(axis : Int, tensors : Array[Tensor]) -> Tensor {
  if tensors.length() == 0 {
    abort("拼接张量不能为空")
  }
  let ref0 = tensors[0]
  if axis < 0 || axis >= ref0.ndim() {
    abort("轴索引越界")
  }
  let mut total_axis = 0
  for i = 0; i < tensors.length(); i = i + 1 {
    let t = tensors[i]
    if t.ndim() != ref0.ndim() {
      abort("张量维度不一致")
    }
    for d = 0; d < ref0.ndim(); d = d + 1 {
      if d != axis && t.shape[d] != ref0.shape[d] {
        abort("张量形状不匹配")
      }
    }
    total_axis = total_axis + t.shape[axis]
  }
  let new_shape = Array::make(ref0.ndim(), 0)
  for d = 0; d < ref0.ndim(); d = d + 1 {
    new_shape[d] = if d == axis { total_axis } else { ref0.shape[d] }
  }
  let out_strides = compute_strides_for_shape(new_shape)
  let mut new_size = 1
  for d = 0; d < new_shape.length(); d = d + 1 {
    new_size = new_size * new_shape[d]
  }
  let new_data = Array::make(new_size, 0.0)
  let mut offset = 0
  for i = 0; i < tensors.length(); i = i + 1 {
    let t = tensors[i]
    for idx = 0; idx < t.size; idx = idx + 1 {
      let old_indices = indices_from_flat(t.shape, t.strides, idx)
      let mut new_idx = 0
      for d = 0; d < t.ndim(); d = d + 1 {
        let v = if d == axis { old_indices[d] + offset } else { old_indices[d] }
        new_idx = new_idx + v * out_strides[d]
      }
      new_data[new_idx] = t.data[idx]
    }
    offset = offset + t.shape[axis]
  }
  Tensor::new(new_data, new_shape)
}

///|
/// 沿指定新轴堆叠
pub fn Tensor::stack(axis : Int, tensors : Array[Tensor]) -> Tensor {
  if tensors.length() == 0 {
    abort("堆叠张量不能为空")
  }
  let ref0 = tensors[0]
  if axis < 0 || axis > ref0.ndim() {
    abort("轴索引越界")
  }
  for i = 0; i < tensors.length(); i = i + 1 {
    let t = tensors[i]
    if t.ndim() != ref0.ndim() {
      abort("张量维度不一致")
    }
    if not(ref0.shapes_equal(t)) {
      abort("张量形状不匹配")
    }
  }
  let new_shape = Array::make(ref0.ndim() + 1, 0)
  let mut old_pos = 0
  for d = 0; d < new_shape.length(); d = d + 1 {
    if d == axis {
      new_shape[d] = tensors.length()
    } else {
      new_shape[d] = ref0.shape[old_pos]
      old_pos = old_pos + 1
    }
  }
  let out_strides = compute_strides_for_shape(new_shape)
  let mut new_size = 1
  for d = 0; d < new_shape.length(); d = d + 1 {
    new_size = new_size * new_shape[d]
  }
  let new_data = Array::make(new_size, 0.0)
  for i = 0; i < tensors.length(); i = i + 1 {
    let t = tensors[i]
    for idx = 0; idx < t.size; idx = idx + 1 {
      let old_indices = indices_from_flat(t.shape, t.strides, idx)
      let new_indices = Array::make(new_shape.length(), 0)
      let mut in_pos = 0
      for d = 0; d < new_shape.length(); d = d + 1 {
        if d == axis {
          new_indices[d] = i
        } else {
          new_indices[d] = old_indices[in_pos]
          in_pos = in_pos + 1
        }
      }
      let mut new_idx = 0
      for d = 0; d < new_shape.length(); d = d + 1 {
        new_idx = new_idx + new_indices[d] * out_strides[d]
      }
      new_data[new_idx] = t.data[idx]
    }
  }
  Tensor::new(new_data, new_shape)
}

///|
/// 展平为1D张量
pub fn Tensor::flatten(self : Tensor) -> Tensor {
  let new_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    new_data[i] = self.data[i]
  }
  Tensor::new(new_data, [self.size])
}

///|
/// 沿指定轴求和
pub fn Tensor::sum_axis(self : Tensor, axis : Int) -> Tensor {
  self.sum_axes([axis])
}

///|
/// 沿指定轴求均值
pub fn Tensor::mean_axis(self : Tensor, axis : Int) -> Tensor {
  self.mean_axes([axis])
}

///|
/// 沿多个轴求和
pub fn Tensor::sum_axes(
  self : Tensor,
  axes : Array[Int],
  keepdim? : Bool = false,
) -> Tensor {
  if axes.length() == 0 {
    return self.copy()
  }
  let ndim = self.ndim()
  let reduce_mask = Array::make(ndim, false)
  for i = 0; i < axes.length(); i = i + 1 {
    let ax = axes[i]
    if ax < 0 || ax >= ndim {
      abort("轴索引越界")
    }
    if reduce_mask[ax] {
      abort("轴重复")
    }
    reduce_mask[ax] = true
  }
  if not(keepdim) && axes.length() == ndim {
    return Tensor::new([self.sum()], [1])
  }
  let out_shape = if keepdim {
    let s = Array::make(ndim, 0)
    for d = 0; d < ndim; d = d + 1 {
      s[d] = if reduce_mask[d] { 1 } else { self.shape[d] }
    }
    s
  } else {
    let s = Array::make(ndim - axes.length(), 0)
    let mut pos = 0
    for d = 0; d < ndim; d = d + 1 {
      if not(reduce_mask[d]) {
        s[pos] = self.shape[d]
        pos = pos + 1
      }
    }
    s
  }
  let out_strides = compute_strides_for_shape(out_shape)
  let mut out_size = 1
  for i = 0; i < out_shape.length(); i = i + 1 {
    out_size = out_size * out_shape[i]
  }
  let result_data = Array::make(out_size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    let coords = indices_from_flat(self.shape, self.strides, i)
    let out_indices = if keepdim {
      let tmp = Array::make(ndim, 0)
      for d = 0; d < ndim; d = d + 1 {
        tmp[d] = if reduce_mask[d] { 0 } else { coords[d] }
      }
      tmp
    } else {
      let tmp = Array::make(out_shape.length(), 0)
      let mut pos = 0
      for d = 0; d < ndim; d = d + 1 {
        if not(reduce_mask[d]) {
          tmp[pos] = coords[d]
          pos = pos + 1
        }
      }
      tmp
    }
    let mut out_idx = 0
    for d = 0; d < out_indices.length(); d = d + 1 {
      out_idx = out_idx + out_indices[d] * out_strides[d]
    }
    result_data[out_idx] = result_data[out_idx] + self.data[i]
  }
  Tensor::new(result_data, out_shape)
}

///|
/// 沿多个轴求均值
pub fn Tensor::mean_axes(
  self : Tensor,
  axes : Array[Int],
  keepdim? : Bool = false,
) -> Tensor {
  if axes.length() == 0 {
    return self.copy()
  }
  let mut denom = 1.0
  for i = 0; i < axes.length(); i = i + 1 {
    let ax = axes[i]
    if ax < 0 || ax >= self.ndim() {
      abort("轴索引越界")
    }
    denom = denom * self.shape[ax].to_double()
  }
  let sum = self.sum_axes(axes, keepdim~)
  sum.scalar_mul(1.0 / denom)
}

///| 统计函数

///|
/// 求和
pub fn Tensor::sum(self : Tensor) -> Double {
  let mut total = 0.0
  for i = 0; i < self.size; i = i + 1 {
    total = total + self.data[i]
  }
  total
}

///|
/// 均值
pub fn Tensor::mean(self : Tensor) -> Double {
  if self.size == 0 {
    abort("空张量没有均值")
  }
  self.sum() / self.size.to_double()
}

///|
/// 最小值
pub fn Tensor::min(self : Tensor) -> Double {
  if self.size == 0 {
    abort("空张量没有最小值")
  }
  let mut min_val = self.data[0]
  for i = 1; i < self.size; i = i + 1 {
    if self.data[i] < min_val {
      min_val = self.data[i]
    }
  }
  min_val
}

///|
/// 最大值
pub fn Tensor::max(self : Tensor) -> Double {
  if self.size == 0 {
    abort("空张量没有最大值")
  }
  let mut max_val = self.data[0]
  for i = 1; i < self.size; i = i + 1 {
    if self.data[i] > max_val {
      max_val = self.data[i]
    }
  }
  max_val
}

///| 实用函数

///|
/// 检查形状是否相等
fn Tensor::shapes_equal(self : Tensor, other : Tensor) -> Bool {
  if self.shape.length() != other.shape.length() {
    return false
  }
  for i = 0; i < self.shape.length(); i = i + 1 {
    if self.shape[i] != other.shape[i] {
      return false
    }
  }
  true
}

///|
/// 计算广播后的形状
fn broadcast_shape(a_shape : Array[Int], b_shape : Array[Int]) -> Array[Int] {
  let a_len = a_shape.length()
  let b_len = b_shape.length()
  let out_len = if a_len > b_len { a_len } else { b_len }
  let out_shape = Array::make(out_len, 0)
  for i = 0; i < out_len; i = i + 1 {
    let a_i = i + a_len - out_len
    let b_i = i + b_len - out_len
    let a_dim = if a_i >= 0 { a_shape[a_i] } else { 1 }
    let b_dim = if b_i >= 0 { b_shape[b_i] } else { 1 }
    if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
      out_shape[i] = if a_dim > b_dim { a_dim } else { b_dim }
    } else {
      abort("张量形状不匹配")
    }
  }
  out_shape
}

///|
/// 广播索引映射
fn broadcast_index(
  out_indices : Array[Int],
  out_shape : Array[Int],
  in_shape : Array[Int],
  in_strides : Array[Int],
) -> Int {
  let out_len = out_shape.length()
  let in_len = in_shape.length()
  let offset = out_len - in_len
  let mut idx = 0
  for i = 0; i < in_len; i = i + 1 {
    let out_i = i + offset
    let v = if in_shape[i] == 1 { 0 } else { out_indices[out_i] }
    idx = idx + v * in_strides[i]
  }
  idx
}

///|
/// 计算形状对应的步长
fn compute_strides_for_shape(shape : Array[Int]) -> Array[Int] {
  let strides = Array::make(shape.length(), 0)
  let mut stride = 1
  for i = shape.length() - 1; i >= 0; i = i - 1 {
    strides[i] = stride
    stride = stride * shape[i]
  }
  strides
}

///|
/// 由扁平索引反推出多维索引
fn indices_from_flat(
  shape : Array[Int],
  strides : Array[Int],
  idx : Int,
) -> Array[Int] {
  let coords = Array::make(shape.length(), 0)
  let mut remain = idx
  for i = 0; i < shape.length(); i = i + 1 {
    coords[i] = remain / strides[i]
    remain = remain % strides[i]
  }
  coords
}

///|
/// 复制张量
pub fn Tensor::copy(self : Tensor) -> Tensor {
  let new_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    new_data[i] = self.data[i]
  }
  Tensor::new(new_data, self.shape)
}

///|
/// 应用函数到每个元素
pub fn Tensor::map(self : Tensor, f : (Double) -> Double) -> Tensor {
  let result_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    result_data[i] = f(self.data[i])
  }
  Tensor::new(result_data, self.shape)
}

///|
/// 转换为 Vector (仅1D张量)
pub fn Tensor::to_vector(self : Tensor) -> Vector {
  if self.ndim() != 1 {
    abort("只有1D张量可以转换为向量")
  }
  Vector::new(self.data)
}

///|
/// 转换为 Matrix (仅2D张量)
pub fn Tensor::to_matrix(self : Tensor) -> Matrix {
  if self.ndim() != 2 {
    abort("只有2D张量可以转换为矩阵")
  }
  let rows = self.shape[0]
  let cols = self.shape[1]
  let mat_data = Array::make(rows, [])
  for i = 0; i < rows; i = i + 1 {
    let row = Array::make(cols, 0.0)
    for j = 0; j < cols; j = j + 1 {
      row[j] = self.data[i * cols + j]
    }
    mat_data[i] = row
  }
  Matrix::new(mat_data)
}

///|
/// 打印张量
pub fn Tensor::to_string(self : Tensor) -> String {
  let mut str = "Tensor(shape=["
  for i = 0; i < self.shape.length(); i = i + 1 {
    str = str + self.shape[i].to_string()
    if i < self.shape.length() - 1 {
      str = str + ", "
    }
  }
  str = str + "], data=[..."

  // 只显示前几个元素
  let display_count = if self.size < 10 { self.size } else { 10 }
  str = str + " "
  for i = 0; i < display_count; i = i + 1 {
    str = str + self.data[i].to_string()
    if i < display_count - 1 {
      str = str + ", "
    }
  }
  if self.size > display_count {
    str = str + ", ..."
  }
  str + "])"
}
