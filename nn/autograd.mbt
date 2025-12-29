///|
/// 简易计算图（Tensor 前向 + 反向传播）

///|
let no_grad_flag : Ref[Bool] = @ref.new(false)

///|
let inference_mode_flag : Ref[Bool] = @ref.new(false)

///|
fn grad_enabled() -> Bool {
  not(no_grad_flag.val) && not(inference_mode_flag.val)
}

///|
pub struct NoGradGuard {
  prev : Bool
}

///|
pub fn no_grad_enter() -> NoGradGuard {
  let prev = no_grad_flag.val
  no_grad_flag.val = true
  { prev, }
}

///|
pub fn no_grad_exit(state : NoGradGuard) -> Unit {
  no_grad_flag.val = state.prev
}

///|
pub fn[T] no_grad(f : () -> T) -> T {
  let state : NoGradGuard = no_grad_enter()
  let out = f()
  no_grad_exit(state)
  out
}

///|
pub struct InferenceModeGuard {
  prev : Bool
}

///|
pub fn inference_mode_enter() -> InferenceModeGuard {
  let prev = inference_mode_flag.val
  inference_mode_flag.val = true
  { prev, }
}

///|
pub fn inference_mode_exit(state : InferenceModeGuard) -> Unit {
  inference_mode_flag.val = state.prev
}

///|
pub fn[T] inference_mode(f : () -> T) -> T {
  let state : InferenceModeGuard = inference_mode_enter()
  let out = f()
  inference_mode_exit(state)
  out
}

///|
pub fn is_grad_enabled() -> Bool {
  grad_enabled()
}

///|
fn compute_strides(shape : Array[Int]) -> Array[Int] {
  let strides = Array::make(shape.length(), 0)
  let mut stride = 1
  for i = shape.length() - 1; i >= 0; i = i - 1 {
    strides[i] = stride
    stride = stride * shape[i]
  }
  strides
}

///|
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
fn tensor_add_inplace(dst : @math.Tensor, src : @math.Tensor) -> Unit {
  let shape = dst.get_shape()
  if shape != src.get_shape() {
    abort("张量形状不一致")
  }
  let strides = compute_strides(shape)
  let total = dst.total_size()
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    dst.set(idx, dst.get(idx) + src.get(idx))
  }
}

///|
fn tensor_hadamard(a : @math.Tensor, b : @math.Tensor) -> @math.Tensor {
  let shape = a.get_shape()
  if shape != b.get_shape() {
    abort("张量形状不一致")
  }
  let strides = compute_strides(shape)
  let total = a.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = a.get(idx) * b.get(idx)
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_add(a : @math.Tensor, b : @math.Tensor) -> @math.Tensor {
  let shape_a = a.get_shape()
  let shape_b = b.get_shape()
  let out_shape = broadcast_shape(shape_a, shape_b)
  let out_strides = compute_strides(out_shape)
  let total = tensor_total_size(out_shape)
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let out_idx = indices_from_flat(out_shape, out_strides, i)
    let a_idx = broadcast_indices(out_idx, out_shape, shape_a)
    let b_idx = broadcast_indices(out_idx, out_shape, shape_b)
    data[i] = a.get(a_idx) + b.get(b_idx)
  }
  @math.Tensor::new(data, out_shape)
}

///|
fn tensor_neg(t : @math.Tensor) -> @math.Tensor {
  let shape = t.get_shape()
  let strides = compute_strides(shape)
  let total = t.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = -t.get(idx)
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_mask_relu(t : @math.Tensor) -> @math.Tensor {
  let shape = t.get_shape()
  let strides = compute_strides(shape)
  let total = t.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    let v = t.get(idx)
    data[i] = if v > 0.0 { 1.0 } else { 0.0 }
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_ones_like(t : @math.Tensor) -> @math.Tensor {
  @math.Tensor::fill(t.get_shape(), 1.0)
}

///|
fn tensor_zeros_like(t : @math.Tensor) -> @math.Tensor {
  @math.Tensor::zeros(t.get_shape())
}

///|
fn tensor_shape_to_tensor(shape : Array[Int]) -> @math.Tensor {
  let data = Array::make(shape.length(), 0.0)
  for i = 0; i < shape.length(); i = i + 1 {
    data[i] = shape[i].to_double()
  }
  @math.Tensor::new(data, [shape.length()])
}

///|
fn tensor_tensor_to_shape(t : @math.Tensor) -> Array[Int] {
  let shape = t.get_shape()
  if shape.length() != 1 {
    abort("shape tensor 必须为 1D")
  }
  let n = shape[0]
  let out = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    out[i] = t.get([i]).to_int()
  }
  out
}

///|
fn tensor_flatten_data(t : @math.Tensor) -> Array[Double] {
  let shape = t.get_shape()
  let strides = compute_strides(shape)
  let total = t.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = t.get(idx)
  }
  data
}

///|
fn tensor_reshape(t : @math.Tensor, shape : Array[Int]) -> @math.Tensor {
  let mut total = 1
  for i = 0; i < shape.length(); i = i + 1 {
    total = total * shape[i]
  }
  if total != t.total_size() {
    abort("reshape 总元素数量不匹配")
  }
  @math.Tensor::new(tensor_flatten_data(t), shape)
}

///|
fn tensor_scale(t : @math.Tensor, scalar : Double) -> @math.Tensor {
  let shape = t.get_shape()
  let strides = compute_strides(shape)
  let total = t.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = t.get(idx) * scalar
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_sub(a : @math.Tensor, b : @math.Tensor) -> @math.Tensor {
  let shape_a = a.get_shape()
  let shape_b = b.get_shape()
  let out_shape = broadcast_shape(shape_a, shape_b)
  let out_strides = compute_strides(out_shape)
  let total = tensor_total_size(out_shape)
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let out_idx = indices_from_flat(out_shape, out_strides, i)
    let a_idx = broadcast_indices(out_idx, out_shape, shape_a)
    let b_idx = broadcast_indices(out_idx, out_shape, shape_b)
    data[i] = a.get(a_idx) - b.get(b_idx)
  }
  @math.Tensor::new(data, out_shape)
}

///|
fn tensor_mul(a : @math.Tensor, b : @math.Tensor) -> @math.Tensor {
  let shape_a = a.get_shape()
  let shape_b = b.get_shape()
  let out_shape = broadcast_shape(shape_a, shape_b)
  let out_strides = compute_strides(out_shape)
  let total = tensor_total_size(out_shape)
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let out_idx = indices_from_flat(out_shape, out_strides, i)
    let a_idx = broadcast_indices(out_idx, out_shape, shape_a)
    let b_idx = broadcast_indices(out_idx, out_shape, shape_b)
    data[i] = a.get(a_idx) * b.get(b_idx)
  }
  @math.Tensor::new(data, out_shape)
}

///|
fn tensor_transpose(t : @math.Tensor, axes : Array[Int]) -> @math.Tensor {
  t.transpose(axes~)
}

///|
fn tensor_matmul(a : @math.Tensor, b : @math.Tensor) -> @math.Tensor {
  let shape_a = a.get_shape()
  let shape_b = b.get_shape()
  if shape_a.length() != 2 || shape_b.length() != 2 {
    abort("matmul 仅支持 2D 张量")
  }
  let m = shape_a[0]
  let k = shape_a[1]
  let kb = shape_b[0]
  let n = shape_b[1]
  if k != kb {
    abort("matmul 维度不匹配")
  }
  let data = Array::make(m * n, 0.0)
  let mut idx = 0
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let mut sum = 0.0
      for p = 0; p < k; p = p + 1 {
        sum = sum + a.get([i, p]) * b.get([p, j])
      }
      data[idx] = sum
      idx = idx + 1
    }
  }
  @math.Tensor::new(data, [m, n])
}

///|
fn tensor_add_bias(x : @math.Tensor, bias : @math.Tensor) -> @math.Tensor {
  let shape_x = x.get_shape()
  let shape_b = bias.get_shape()
  if shape_x.length() != 2 || shape_b.length() != 1 {
    abort("add_bias 需要形状 [batch, features] + [features]")
  }
  let n = shape_x[0]
  let m = shape_x[1]
  if shape_b[0] != m {
    abort("bias 维度不匹配")
  }
  let data = Array::make(n * m, 0.0)
  let mut idx = 0
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      data[idx] = x.get([i, j]) + bias.get([j])
      idx = idx + 1
    }
  }
  @math.Tensor::new(data, [n, m])
}

///|
fn tensor_sum_all(x : @math.Tensor) -> @math.Tensor {
  let shape = x.get_shape()
  let strides = compute_strides(shape)
  let total = x.total_size()
  let mut sum = 0.0
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    sum = sum + x.get(idx)
  }
  @math.Tensor::new([sum], [1])
}

///|
fn tensor_total_size(shape : Array[Int]) -> Int {
  let mut total = 1
  for i = 0; i < shape.length(); i = i + 1 {
    total = total * shape[i]
  }
  total
}

///|
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
      abort("张量形状不一致")
    }
  }
  out_shape
}

///|
fn broadcast_indices(
  out_indices : Array[Int],
  out_shape : Array[Int],
  in_shape : Array[Int],
) -> Array[Int] {
  let out_len = out_shape.length()
  let in_len = in_shape.length()
  let offset = out_len - in_len
  let idx = Array::make(in_len, 0)
  for i = 0; i < in_len; i = i + 1 {
    let out_i = i + offset
    idx[i] = if in_shape[i] == 1 { 0 } else { out_indices[out_i] }
  }
  idx
}

///|
fn tensor_reduce_to_shape(
  grad : @math.Tensor,
  shape : Array[Int],
) -> @math.Tensor {
  let grad_shape = grad.get_shape()
  if grad_shape == shape {
    return @math.Tensor::new(tensor_flatten_data(grad), shape)
  }
  if grad_shape.length() < shape.length() {
    abort("reduce 目标形状不合法")
  }
  let out = @math.Tensor::zeros(shape)
  let grad_strides = compute_strides(grad_shape)
  let total = grad.total_size()
  let offset = grad_shape.length() - shape.length()
  for i = 0; i < total; i = i + 1 {
    let g_idx = indices_from_flat(grad_shape, grad_strides, i)
    let out_idx = Array::make(shape.length(), 0)
    for d = 0; d < shape.length(); d = d + 1 {
      let g_pos = d + offset
      let v = if shape[d] == 1 { 0 } else { g_idx[g_pos] }
      out_idx[d] = v
    }
    out.set(out_idx, out.get(out_idx) + grad.get(g_idx))
  }
  out
}

///|
fn tensor_broadcast_to(x : @math.Tensor, shape : Array[Int]) -> @math.Tensor {
  let in_shape = x.get_shape()
  let out_shape = broadcast_shape(in_shape, shape)
  if out_shape != shape {
    abort("广播目标形状不匹配")
  }
  let out_strides = compute_strides(out_shape)
  let total = tensor_total_size(out_shape)
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let out_idx = indices_from_flat(out_shape, out_strides, i)
    let in_idx = broadcast_indices(out_idx, out_shape, in_shape)
    data[i] = x.get(in_idx)
  }
  @math.Tensor::new(data, out_shape)
}

///|
fn tensor_expand_for_axis(
  t : @math.Tensor,
  input_shape : Array[Int],
  axis : Int,
  keepdim : Bool,
) -> @math.Tensor {
  let shape = if keepdim {
    t
  } else {
    let reshape_shape = Array::make(input_shape.length(), 0)
    for i = 0; i < input_shape.length(); i = i + 1 {
      reshape_shape[i] = if i == axis { 1 } else { input_shape[i] }
    }
    tensor_reshape(t, reshape_shape)
  }
  tensor_broadcast_to(shape, input_shape)
}

///|
fn tensor_sum_axes(
  x : @math.Tensor,
  axes : Array[Int],
  keepdim? : Bool = false,
) -> @math.Tensor {
  if axes.length() == 0 {
    return @math.Tensor::new(tensor_flatten_data(x), x.get_shape())
  }
  let shape = x.get_shape()
  let ndim = shape.length()
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
  let out_shape = if keepdim {
    let s = Array::make(ndim, 0)
    for d = 0; d < ndim; d = d + 1 {
      s[d] = if reduce_mask[d] { 1 } else { shape[d] }
    }
    s
  } else {
    let s = Array::make(ndim - axes.length(), 0)
    let mut pos = 0
    for d = 0; d < ndim; d = d + 1 {
      if not(reduce_mask[d]) {
        s[pos] = shape[d]
        pos = pos + 1
      }
    }
    s
  }
  let out_strides = compute_strides(out_shape)
  let out_size = tensor_total_size(out_shape)
  let data = Array::make(out_size, 0.0)
  let strides = compute_strides(shape)
  let total = x.total_size()
  for i = 0; i < total; i = i + 1 {
    let coords = indices_from_flat(shape, strides, i)
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
    data[out_idx] = data[out_idx] + x.get(coords)
  }
  @math.Tensor::new(data, out_shape)
}

///|
fn tensor_mean_axes(
  x : @math.Tensor,
  axes : Array[Int],
  keepdim? : Bool = false,
) -> @math.Tensor {
  if axes.length() == 0 {
    return @math.Tensor::new(tensor_flatten_data(x), x.get_shape())
  }
  let mut denom = 1.0
  let shape = x.get_shape()
  for i = 0; i < axes.length(); i = i + 1 {
    let ax = axes[i]
    if ax < 0 || ax >= shape.length() {
      abort("轴索引越界")
    }
    denom = denom * shape[ax].to_double()
  }
  let sum = tensor_sum_axes(x, axes, keepdim~)
  tensor_scale(sum, 1.0 / denom)
}

///|
fn tensor_logsumexp_axis(
  x : @math.Tensor,
  axis : Int,
  keepdim? : Bool = false,
) -> @math.Tensor {
  let shape = x.get_shape()
  let ndim = shape.length()
  if axis < 0 || axis >= ndim {
    abort("轴索引越界")
  }
  let reduce_mask = Array::make(ndim, false)
  reduce_mask[axis] = true
  let out_shape = if keepdim {
    let s = Array::make(ndim, 0)
    for d = 0; d < ndim; d = d + 1 {
      s[d] = if reduce_mask[d] { 1 } else { shape[d] }
    }
    s
  } else {
    let s = Array::make(ndim - 1, 0)
    let mut pos = 0
    for d = 0; d < ndim; d = d + 1 {
      if not(reduce_mask[d]) {
        s[pos] = shape[d]
        pos = pos + 1
      }
    }
    s
  }
  let out_strides = compute_strides(out_shape)
  let out_size = tensor_total_size(out_shape)
  let max_data = Array::make(out_size, -@double.infinity)
  let sum_data = Array::make(out_size, 0.0)
  let strides = compute_strides(shape)
  let total = x.total_size()
  for i = 0; i < total; i = i + 1 {
    let coords = indices_from_flat(shape, strides, i)
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
    let v = x.get(coords)
    if v > max_data[out_idx] {
      max_data[out_idx] = v
    }
  }
  for i = 0; i < total; i = i + 1 {
    let coords = indices_from_flat(shape, strides, i)
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
    sum_data[out_idx] = sum_data[out_idx] +
      @moonbitlang/core/math.exp(x.get(coords) - max_data[out_idx])
  }
  let out_data = Array::make(out_size, 0.0)
  for i = 0; i < out_size; i = i + 1 {
    out_data[i] = @moonbitlang/core/math.ln(sum_data[i]) + max_data[i]
  }
  @math.Tensor::new(out_data, out_shape)
}

///|
fn tensor_exp(x : @math.Tensor) -> @math.Tensor {
  let shape = x.get_shape()
  let strides = compute_strides(shape)
  let total = x.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = @moonbitlang/core/math.exp(x.get(idx))
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_sign(x : @math.Tensor) -> @math.Tensor {
  let shape = x.get_shape()
  let strides = compute_strides(shape)
  let total = x.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    let v = x.get(idx)
    data[i] = if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_dropout(
  x : @math.Tensor,
  rate : Double,
  seed : Int,
) -> (@math.Tensor, @math.Tensor, Int) {
  let shape = x.get_shape()
  let strides = compute_strides(shape)
  let total = x.total_size()
  let data = Array::make(total, 0.0)
  let mask = Array::make(total, 0.0)
  let keep = 1.0 - rate
  let scale = 1.0 / keep
  let mut rng = seed
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    rng = (1103515245 * rng + 12345) & 0x7fffffff
    let r = rng.to_double() / (0x7fffffff).to_double()
    if r < keep {
      let v = x.get(idx) * scale
      data[i] = v
      mask[i] = scale
    } else {
      data[i] = 0.0
      mask[i] = 0.0
    }
  }
  (@math.Tensor::new(data, shape), @math.Tensor::new(mask, shape), rng)
}

///|
fn tensor_dropout2d(
  x : @math.Tensor,
  rate : Double,
  seed : Int,
) -> (@math.Tensor, @math.Tensor, Int) {
  let shape = x.get_shape()
  if shape.length() != 4 {
    abort("dropout2d 仅支持 4D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  let h = shape[2]
  let w = shape[3]
  let data = Array::make(n * c * h * w, 0.0)
  let mask = Array::make(n * c * h * w, 0.0)
  let keep = 1.0 - rate
  let scale = 1.0 / keep
  let mut rng = seed
  let mut idx = 0
  for ni = 0; ni < n; ni = ni + 1 {
    for ci = 0; ci < c; ci = ci + 1 {
      rng = (1103515245 * rng + 12345) & 0x7fffffff
      let r = rng.to_double() / (0x7fffffff).to_double()
      let keep_channel = r < keep
      for hi = 0; hi < h; hi = hi + 1 {
        for wi = 0; wi < w; wi = wi + 1 {
          let v = x.get([ni, ci, hi, wi])
          if keep_channel {
            data[idx] = v * scale
            mask[idx] = scale
          } else {
            data[idx] = 0.0
            mask[idx] = 0.0
          }
          idx = idx + 1
        }
      }
    }
  }
  (@math.Tensor::new(data, shape), @math.Tensor::new(mask, shape), rng)
}

///|
fn tensor_layernorm_forward(
  x : @math.Tensor,
  gamma : @math.Tensor,
  beta : @math.Tensor,
  eps : Double,
) -> (@math.Tensor, @math.Tensor, @math.Tensor) {
  let shape = x.get_shape()
  if shape.length() != 2 {
    abort("layernorm 仅支持 2D 张量")
  }
  let n = shape[0]
  let m = shape[1]
  if gamma.get_shape() != [m] || beta.get_shape() != [m] {
    abort("layernorm 维度不匹配")
  }
  let data = Array::make(n * m, 0.0)
  let mean = Array::make(n, 0.0)
  let variance = Array::make(n, 0.0)
  let mut idx = 0
  for i = 0; i < n; i = i + 1 {
    let mut sum = 0.0
    for j = 0; j < m; j = j + 1 {
      sum = sum + x.get([i, j])
    }
    let mu = sum / m.to_double()
    mean[i] = mu
    let mut var_val = 0.0
    for j = 0; j < m; j = j + 1 {
      let diff = x.get([i, j]) - mu
      var_val = var_val + diff * diff
    }
    var_val = var_val / m.to_double()
    variance[i] = var_val
    let denom = (var_val + eps).sqrt()
    for j = 0; j < m; j = j + 1 {
      let norm = (x.get([i, j]) - mu) / denom
      data[idx] = norm * gamma.get([j]) + beta.get([j])
      idx = idx + 1
    }
  }
  (
    @math.Tensor::new(data, [n, m]),
    @math.Tensor::new(mean, [n]),
    @math.Tensor::new(variance, [n]),
  )
}

///|
fn tensor_conv2d(
  x : @math.Tensor,
  w : @math.Tensor,
  b : @math.Tensor,
  stride : Int,
  padding : Int,
) -> @math.Tensor {
  let shape_x = x.get_shape()
  let shape_w = w.get_shape()
  if shape_x.length() != 4 || shape_w.length() != 4 {
    abort("conv2d 仅支持 4D 张量")
  }
  let n = shape_x[0]
  let cin = shape_x[1]
  let h = shape_x[2]
  let wi = shape_x[3]
  let cout = shape_w[0]
  let cin_w = shape_w[1]
  let kh = shape_w[2]
  let kw = shape_w[3]
  if cin != cin_w {
    abort("conv2d 输入通道不匹配")
  }
  if b.get_shape() != [cout] {
    abort("conv2d bias 形状不匹配")
  }
  let out_h = (h + 2 * padding - kh) / stride + 1
  let out_w = (wi + 2 * padding - kw) / stride + 1
  let out = @math.Tensor::zeros([n, cout, out_h, out_w])
  for ni = 0; ni < n; ni = ni + 1 {
    for co = 0; co < cout; co = co + 1 {
      for oh = 0; oh < out_h; oh = oh + 1 {
        for ow = 0; ow < out_w; ow = ow + 1 {
          let mut sum = b.get([co])
          for ci = 0; ci < cin; ci = ci + 1 {
            for khi = 0; khi < kh; khi = khi + 1 {
              for kwi = 0; kwi < kw; kwi = kwi + 1 {
                let ih = oh * stride + khi - padding
                let iw = ow * stride + kwi - padding
                if ih >= 0 && ih < h && iw >= 0 && iw < wi {
                  sum = sum +
                    x.get([ni, ci, ih, iw]) * w.get([co, ci, khi, kwi])
                }
              }
            }
          }
          out.set([ni, co, oh, ow], sum)
        }
      }
    }
  }
  out
}

///|
fn tensor_maxpool2d(
  x : @math.Tensor,
  kernel : Int,
  stride : Int,
) -> (@math.Tensor, @math.Tensor, @math.Tensor) {
  let shape = x.get_shape()
  if shape.length() != 4 {
    abort("maxpool2d 仅支持 4D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  let h = shape[2]
  let w = shape[3]
  let out_h = (h - kernel) / stride + 1
  let out_w = (w - kernel) / stride + 1
  let out = @math.Tensor::zeros([n, c, out_h, out_w])
  let idx_h = @math.Tensor::zeros([n, c, out_h, out_w])
  let idx_w = @math.Tensor::zeros([n, c, out_h, out_w])
  for ni = 0; ni < n; ni = ni + 1 {
    for ci = 0; ci < c; ci = ci + 1 {
      for oh = 0; oh < out_h; oh = oh + 1 {
        for ow = 0; ow < out_w; ow = ow + 1 {
          let mut max_v = -@double.infinity
          let mut max_h = 0
          let mut max_w = 0
          for khi = 0; khi < kernel; khi = khi + 1 {
            for kwi = 0; kwi < kernel; kwi = kwi + 1 {
              let ih = oh * stride + khi
              let iw = ow * stride + kwi
              let v = x.get([ni, ci, ih, iw])
              if v > max_v {
                max_v = v
                max_h = ih
                max_w = iw
              }
            }
          }
          out.set([ni, ci, oh, ow], max_v)
          idx_h.set([ni, ci, oh, ow], max_h.to_double())
          idx_w.set([ni, ci, oh, ow], max_w.to_double())
        }
      }
    }
  }
  (out, idx_h, idx_w)
}

///|
fn tensor_avgpool2d(
  x : @math.Tensor,
  kernel : Int,
  stride : Int,
) -> @math.Tensor {
  let shape = x.get_shape()
  if shape.length() != 4 {
    abort("avgpool2d 仅支持 4D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  let h = shape[2]
  let w = shape[3]
  let out_h = (h - kernel) / stride + 1
  let out_w = (w - kernel) / stride + 1
  let out = @math.Tensor::zeros([n, c, out_h, out_w])
  let denom = (kernel * kernel).to_double()
  for ni = 0; ni < n; ni = ni + 1 {
    for ci = 0; ci < c; ci = ci + 1 {
      for oh = 0; oh < out_h; oh = oh + 1 {
        for ow = 0; ow < out_w; ow = ow + 1 {
          let mut sum = 0.0
          for khi = 0; khi < kernel; khi = khi + 1 {
            for kwi = 0; kwi < kernel; kwi = kwi + 1 {
              let ih = oh * stride + khi
              let iw = ow * stride + kwi
              sum = sum + x.get([ni, ci, ih, iw])
            }
          }
          out.set([ni, ci, oh, ow], sum / denom)
        }
      }
    }
  }
  out
}

///|
fn tensor_softmax_cross_entropy(
  logits : @math.Tensor,
  labels : Array[Int],
) -> (@math.Tensor, @math.Tensor) {
  let shape = logits.get_shape()
  if shape.length() != 2 {
    abort("softmax_cross_entropy 仅支持 2D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  if labels.length() != n {
    abort("标签长度不匹配")
  }
  let probs_data = Array::make(n * c, 0.0)
  let mut loss = 0.0
  for i = 0; i < n; i = i + 1 {
    let cls = labels[i]
    if cls < 0 || cls >= c {
      abort("类别索引越界")
    }
    let mut max_v = logits.get([i, 0])
    for j = 1; j < c; j = j + 1 {
      let v = logits.get([i, j])
      if v > max_v {
        max_v = v
      }
    }
    let mut sum = 0.0
    for j = 0; j < c; j = j + 1 {
      let v = @moonbitlang/core/math.exp(logits.get([i, j]) - max_v)
      sum = sum + v
      probs_data[i * c + j] = v
    }
    for j = 0; j < c; j = j + 1 {
      probs_data[i * c + j] = probs_data[i * c + j] / sum
    }
    let mut logprob = logits.get([i, cls]) -
      max_v -
      @moonbitlang/core/math.ln(sum)
    if logprob < -1.0e9 {
      logprob = -1.0e9
    }
    if logprob > 1.0e9 {
      logprob = 1.0e9
    }
    loss = loss - logprob
  }
  let loss_val = loss / n.to_double()
  (@math.Tensor::new([loss_val], [1]), @math.Tensor::new(probs_data, [n, c]))
}

///|
fn labels_to_tensor(labels : Array[Int]) -> @math.Tensor {
  let data = Array::make(labels.length(), 0.0)
  for i = 0; i < labels.length(); i = i + 1 {
    data[i] = labels[i].to_double()
  }
  @math.Tensor::new(data, [labels.length()])
}

///|
pub enum Op {
  Input
  Add
  Sub
  Mul
  Relu
  MatMul
  AddBias
  Sum
  SumAxis
  MeanAxis
  Dropout
  Dropout2d
  BatchNorm1d
  BatchNorm2d
  LayerNorm
  Embedding
  Conv2d
  MaxPool2d
  AvgPool2d
  Flatten2d
  SoftmaxCrossEntropy
  NLLLoss
  MSELoss
  MAELoss
  Reshape
  Transpose
  LogSoftmax
  LogSumExp
}

///|
pub struct Node {
  value : @math.Tensor
  mut grad : @math.Tensor
  op : Op
  parents : Array[Int]
  requires_grad : Bool
  aux_tensors : Array[@math.Tensor]
  aux_scalars : Array[Double]
  aux_bool : Bool
}

///|
pub struct ComputationGraph {
  nodes : Array[Node]
}

///|
pub fn ComputationGraph::new() -> ComputationGraph {
  { nodes: [] }
}

///|
/// 简易 tensor 风格封装
pub struct AutoTensor {
  graph : ComputationGraph
  id : Int
}

///|
pub fn AutoTensor::new(
  value : @math.Tensor,
  requires_grad? : Bool = true,
) -> AutoTensor {
  let g = ComputationGraph::new()
  let id = g.add_input(value, requires_grad~)
  { graph: g, id }
}

///|
/// 基于已有张量的计算图创建新的输入
pub fn AutoTensor::from_other(
  other : AutoTensor,
  value : @math.Tensor,
  requires_grad? : Bool = true,
) -> AutoTensor {
  let g = other.graph
  let id = g.add_input(value, requires_grad~)
  { graph: g, id }
}

///|
pub fn AutoTensor::value(self : AutoTensor) -> @math.Tensor {
  self.graph.value_of(self.id)
}

///|
pub fn AutoTensor::grad(self : AutoTensor) -> @math.Tensor {
  self.graph.grad_of(self.id)
}

///|
pub fn AutoTensor::backward(self : AutoTensor) -> Unit {
  self.graph.backward(self.id)
}

///|
pub fn AutoTensor::backward_with_grad(
  self : AutoTensor,
  grad : @math.Tensor,
) -> Unit {
  self.graph.backward_with_grad(self.id, grad)
}

///|
pub fn AutoTensor::add(self : AutoTensor, other : AutoTensor) -> AutoTensor {
  let g = self.graph
  let id = g.add(self.id, other.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::sub(self : AutoTensor, other : AutoTensor) -> AutoTensor {
  let g = self.graph
  let id = g.sub(self.id, other.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::mul(self : AutoTensor, other : AutoTensor) -> AutoTensor {
  let g = self.graph
  let id = g.mul(self.id, other.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::relu(self : AutoTensor) -> AutoTensor {
  let g = self.graph
  let id = g.relu(self.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::matmul(self : AutoTensor, other : AutoTensor) -> AutoTensor {
  let g = self.graph
  let id = g.matmul(self.id, other.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::add_bias(self : AutoTensor, bias : AutoTensor) -> AutoTensor {
  let g = self.graph
  let id = g.add_bias(self.id, bias.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::sum(self : AutoTensor) -> AutoTensor {
  let g = self.graph
  let id = g.sum(self.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::sum_axis(
  self : AutoTensor,
  axis : Int,
  keepdim? : Bool = false,
) -> AutoTensor {
  let g = self.graph
  let id = g.sum_axis(self.id, axis, keepdim~)
  { graph: g, id }
}

///|
pub fn AutoTensor::mean_axis(
  self : AutoTensor,
  axis : Int,
  keepdim? : Bool = false,
) -> AutoTensor {
  let g = self.graph
  let id = g.mean_axis(self.id, axis, keepdim~)
  { graph: g, id }
}

///|
pub fn AutoTensor::dropout(
  self : AutoTensor,
  rate : Double,
  seed : Int,
  train? : Bool = true,
) -> (AutoTensor, Int) {
  if not(train) || rate == 0.0 {
    return (self, seed)
  }
  let g = self.graph
  let (id, new_seed) = g.dropout(self.id, rate, seed)
  ({ graph: g, id }, new_seed)
}

///|
pub fn AutoTensor::dropout2d(
  self : AutoTensor,
  rate : Double,
  seed : Int,
  train? : Bool = true,
) -> (AutoTensor, Int) {
  if not(train) || rate == 0.0 {
    return (self, seed)
  }
  let g = self.graph
  let (id, new_seed) = g.dropout2d(self.id, rate, seed)
  ({ graph: g, id }, new_seed)
}

///|
pub fn AutoTensor::batchnorm1d_cached(
  self : AutoTensor,
  out : @math.Tensor,
  mean : @math.Tensor,
  variance : @math.Tensor,
  gamma : AutoTensor,
  beta : AutoTensor,
  eps : Double,
  train? : Bool = true,
) -> AutoTensor {
  let g = self.graph
  let id = g.batchnorm1d_cached(
    self.id,
    gamma.id,
    beta.id,
    out,
    mean,
    variance,
    eps,
    train~,
  )
  { graph: g, id }
}

///|
pub fn AutoTensor::batchnorm2d_cached(
  self : AutoTensor,
  out : @math.Tensor,
  mean : @math.Tensor,
  variance : @math.Tensor,
  gamma : AutoTensor,
  beta : AutoTensor,
  eps : Double,
  train? : Bool = true,
) -> AutoTensor {
  let g = self.graph
  let id = g.batchnorm2d_cached(
    self.id,
    gamma.id,
    beta.id,
    out,
    mean,
    variance,
    eps,
    train~,
  )
  { graph: g, id }
}

///|
pub fn AutoTensor::layernorm(
  self : AutoTensor,
  gamma : AutoTensor,
  beta : AutoTensor,
  eps : Double,
) -> AutoTensor {
  let g = self.graph
  let id = g.layernorm(self.id, gamma.id, beta.id, eps)
  { graph: g, id }
}

///|
pub fn AutoTensor::embedding(
  self : AutoTensor,
  weight : AutoTensor,
) -> AutoTensor {
  let g = self.graph
  let id = g.embedding(weight.id, self.value())
  { graph: g, id }
}

///|
pub fn AutoTensor::conv2d(
  self : AutoTensor,
  weight : AutoTensor,
  bias : AutoTensor,
  stride : Int,
  padding : Int,
) -> AutoTensor {
  let g = self.graph
  let id = g.conv2d(self.id, weight.id, bias.id, stride, padding)
  { graph: g, id }
}

///|
pub fn AutoTensor::maxpool2d(
  self : AutoTensor,
  kernel : Int,
  stride : Int,
) -> AutoTensor {
  let g = self.graph
  let id = g.maxpool2d(self.id, kernel, stride)
  { graph: g, id }
}

///|
pub fn AutoTensor::avgpool2d(
  self : AutoTensor,
  kernel : Int,
  stride : Int,
) -> AutoTensor {
  let g = self.graph
  let id = g.avgpool2d(self.id, kernel, stride)
  { graph: g, id }
}

///|
pub fn AutoTensor::flatten2d(self : AutoTensor) -> AutoTensor {
  let g = self.graph
  let id = g.flatten2d(self.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::reshape(self : AutoTensor, shape : Array[Int]) -> AutoTensor {
  let g = self.graph
  let id = g.reshape(self.id, shape)
  { graph: g, id }
}

///|
pub fn AutoTensor::transpose(
  self : AutoTensor,
  axes? : Array[Int] = [],
) -> AutoTensor {
  let g = self.graph
  let id = g.transpose(self.id, axes)
  { graph: g, id }
}

///|
pub fn AutoTensor::permute(self : AutoTensor, axes : Array[Int]) -> AutoTensor {
  self.transpose(axes~)
}

///|
pub fn AutoTensor::softmax_cross_entropy(
  self : AutoTensor,
  y_true : Array[Int],
) -> AutoTensor {
  let g = self.graph
  let id = g.softmax_cross_entropy(self.id, y_true)
  { graph: g, id }
}

///|
pub fn AutoTensor::cross_entropy(
  self : AutoTensor,
  y_true : Array[Int],
) -> AutoTensor {
  self.softmax_cross_entropy(y_true)
}

///|
pub fn AutoTensor::nll_loss(
  self : AutoTensor,
  y_true : Array[Int],
) -> AutoTensor {
  let g = self.graph
  let id = g.nll_loss(self.id, y_true)
  { graph: g, id }
}

///|
pub fn AutoTensor::mse_loss(
  self : AutoTensor,
  target : AutoTensor,
) -> AutoTensor {
  let g = self.graph
  let id = g.mse_loss(self.id, target.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::mae_loss(
  self : AutoTensor,
  target : AutoTensor,
) -> AutoTensor {
  let g = self.graph
  let id = g.mae_loss(self.id, target.id)
  { graph: g, id }
}

///|
pub fn AutoTensor::log_softmax(
  self : AutoTensor,
  axis? : Int = 1,
) -> AutoTensor {
  let g = self.graph
  let id = g.log_softmax(self.id, axis)
  { graph: g, id }
}

///|
pub fn AutoTensor::logsumexp(
  self : AutoTensor,
  axis? : Int = 1,
  keepdim? : Bool = false,
) -> AutoTensor {
  let g = self.graph
  let id = g.logsumexp(self.id, axis, keepdim~)
  { graph: g, id }
}

///|
/// 运算符重载
pub impl Add for AutoTensor with add(self, other) {
  self.add(other)
}

///|
pub impl Sub for AutoTensor with sub(self, other) {
  self.sub(other)
}

///|
pub impl Mul for AutoTensor with mul(self, other) {
  self.mul(other)
}

///|
pub fn ComputationGraph::add_input(
  self : ComputationGraph,
  value : @math.Tensor,
  requires_grad? : Bool = true,
) -> Int {
  let grad = tensor_zeros_like(value)
  self.nodes.push({
    value,
    grad,
    op: Input,
    parents: [],
    requires_grad,
    aux_tensors: [],
    aux_scalars: [],
    aux_bool: false,
  })
  self.nodes.length() - 1
}

///|
fn ComputationGraph::record_op(
  self : ComputationGraph,
  value : @math.Tensor,
  op : Op,
  parents : Array[Int],
  aux_tensors : Array[@math.Tensor],
  aux_scalars : Array[Double],
  aux_bool : Bool,
) -> Int {
  if not(grad_enabled()) {
    return self.add_input(value, requires_grad=false)
  }
  let grad = tensor_zeros_like(value)
  self.nodes.push({
    value,
    grad,
    op,
    parents,
    requires_grad: true,
    aux_tensors,
    aux_scalars,
    aux_bool,
  })
  self.nodes.length() - 1
}

///|
pub fn ComputationGraph::add(self : ComputationGraph, a : Int, b : Int) -> Int {
  let va = self.nodes[a].value
  let vb = self.nodes[b].value
  let out = tensor_add(va, vb)
  self.record_op(out, Add, [a, b], [], [], false)
}

///|
pub fn ComputationGraph::sub(self : ComputationGraph, a : Int, b : Int) -> Int {
  let va = self.nodes[a].value
  let vb = self.nodes[b].value
  let out = tensor_sub(va, vb)
  self.record_op(out, Sub, [a, b], [], [], false)
}

///|
pub fn ComputationGraph::mul(self : ComputationGraph, a : Int, b : Int) -> Int {
  let va = self.nodes[a].value
  let vb = self.nodes[b].value
  let out = tensor_mul(va, vb)
  self.record_op(out, Mul, [a, b], [], [], false)
}

///|
pub fn ComputationGraph::relu(self : ComputationGraph, a : Int) -> Int {
  let va = self.nodes[a].value
  let shape = va.get_shape()
  let strides = compute_strides(shape)
  let total = va.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    let v = va.get(idx)
    data[i] = if v > 0.0 { v } else { 0.0 }
  }
  let out = @math.Tensor::new(data, shape)
  self.record_op(out, Relu, [a], [], [], false)
}

///|
pub fn ComputationGraph::matmul(
  self : ComputationGraph,
  a : Int,
  b : Int,
) -> Int {
  let va = self.nodes[a].value
  let vb = self.nodes[b].value
  let out = tensor_matmul(va, vb)
  self.record_op(out, MatMul, [a, b], [], [], false)
}

///|
pub fn ComputationGraph::add_bias(
  self : ComputationGraph,
  x : Int,
  bias : Int,
) -> Int {
  let vx = self.nodes[x].value
  let vb = self.nodes[bias].value
  let out = tensor_add_bias(vx, vb)
  self.record_op(out, AddBias, [x, bias], [], [], false)
}

///|
pub fn ComputationGraph::sum(self : ComputationGraph, x : Int) -> Int {
  let vx = self.nodes[x].value
  let out = tensor_sum_all(vx)
  self.record_op(out, Sum, [x], [], [], false)
}

///|
pub fn ComputationGraph::sum_axis(
  self : ComputationGraph,
  x : Int,
  axis : Int,
  keepdim? : Bool = false,
) -> Int {
  let vx = self.nodes[x].value
  let out = tensor_sum_axes(vx, [axis], keepdim~)
  self.record_op(
    out,
    SumAxis,
    [x],
    [tensor_shape_to_tensor(vx.get_shape())],
    [axis.to_double()],
    keepdim,
  )
}

///|
pub fn ComputationGraph::mean_axis(
  self : ComputationGraph,
  x : Int,
  axis : Int,
  keepdim? : Bool = false,
) -> Int {
  let vx = self.nodes[x].value
  let out = tensor_mean_axes(vx, [axis], keepdim~)
  self.record_op(
    out,
    MeanAxis,
    [x],
    [tensor_shape_to_tensor(vx.get_shape())],
    [axis.to_double()],
    keepdim,
  )
}

///|
pub fn ComputationGraph::dropout(
  self : ComputationGraph,
  x : Int,
  rate : Double,
  seed : Int,
) -> (Int, Int) {
  let vx = self.nodes[x].value
  let (out, mask, new_seed) = tensor_dropout(vx, rate, seed)
  let id = self.record_op(out, Dropout, [x], [mask], [], false)
  (id, new_seed)
}

///|
pub fn ComputationGraph::dropout2d(
  self : ComputationGraph,
  x : Int,
  rate : Double,
  seed : Int,
) -> (Int, Int) {
  let vx = self.nodes[x].value
  let (out, mask, new_seed) = tensor_dropout2d(vx, rate, seed)
  let id = self.record_op(out, Dropout2d, [x], [mask], [], false)
  (id, new_seed)
}

///|
pub fn ComputationGraph::batchnorm1d_cached(
  self : ComputationGraph,
  x : Int,
  gamma : Int,
  beta : Int,
  out : @math.Tensor,
  mean : @math.Tensor,
  variance : @math.Tensor,
  eps : Double,
  train? : Bool = true,
) -> Int {
  self.record_op(
    out,
    BatchNorm1d,
    [x, gamma, beta],
    [mean, variance],
    [eps],
    train,
  )
}

///|
pub fn ComputationGraph::batchnorm2d_cached(
  self : ComputationGraph,
  x : Int,
  gamma : Int,
  beta : Int,
  out : @math.Tensor,
  mean : @math.Tensor,
  variance : @math.Tensor,
  eps : Double,
  train? : Bool = true,
) -> Int {
  self.record_op(
    out,
    BatchNorm2d,
    [x, gamma, beta],
    [mean, variance],
    [eps],
    train,
  )
}

///|
pub fn ComputationGraph::layernorm(
  self : ComputationGraph,
  x : Int,
  gamma : Int,
  beta : Int,
  eps : Double,
) -> Int {
  let vx = self.nodes[x].value
  let g = self.nodes[gamma].value
  let b = self.nodes[beta].value
  let (out, mean, variance) = tensor_layernorm_forward(vx, g, b, eps)
  self.record_op(
    out,
    LayerNorm,
    [x, gamma, beta],
    [mean, variance],
    [eps],
    false,
  )
}

///|
pub fn ComputationGraph::embedding(
  self : ComputationGraph,
  weight : Int,
  indices : @math.Tensor,
) -> Int {
  let w = self.nodes[weight].value
  let shape_w = w.get_shape()
  if shape_w.length() != 2 {
    abort("embedding 权重需要 2D 张量")
  }
  let n = indices.total_size()
  let dim = shape_w[1]
  let data = Array::make(n * dim, 0.0)
  let mut idx = 0
  for i = 0; i < n; i = i + 1 {
    let row = indices.get([i]).to_int()
    if row < 0 || row >= shape_w[0] {
      abort("embedding 索引越界")
    }
    for j = 0; j < dim; j = j + 1 {
      data[idx] = w.get([row, j])
      idx = idx + 1
    }
  }
  let out = @math.Tensor::new(data, [n, dim])
  self.record_op(out, Embedding, [weight], [indices], [], false)
}

///|
pub fn ComputationGraph::conv2d(
  self : ComputationGraph,
  x : Int,
  weight : Int,
  bias : Int,
  stride : Int,
  padding : Int,
) -> Int {
  let vx = self.nodes[x].value
  let vw = self.nodes[weight].value
  let vb = self.nodes[bias].value
  let out = tensor_conv2d(vx, vw, vb, stride, padding)
  self.record_op(
    out,
    Conv2d,
    [x, weight, bias],
    [],
    [stride.to_double(), padding.to_double()],
    false,
  )
}

///|
pub fn ComputationGraph::maxpool2d(
  self : ComputationGraph,
  x : Int,
  kernel : Int,
  stride : Int,
) -> Int {
  let vx = self.nodes[x].value
  let (out, idx_h, idx_w) = tensor_maxpool2d(vx, kernel, stride)
  self.record_op(
    out,
    MaxPool2d,
    [x],
    [idx_h, idx_w],
    [kernel.to_double(), stride.to_double()],
    false,
  )
}

///|
pub fn ComputationGraph::avgpool2d(
  self : ComputationGraph,
  x : Int,
  kernel : Int,
  stride : Int,
) -> Int {
  let vx = self.nodes[x].value
  let out = tensor_avgpool2d(vx, kernel, stride)
  self.record_op(
    out,
    AvgPool2d,
    [x],
    [],
    [kernel.to_double(), stride.to_double()],
    false,
  )
}

///|
pub fn ComputationGraph::flatten2d(self : ComputationGraph, x : Int) -> Int {
  let vx = self.nodes[x].value
  let shape = vx.get_shape()
  if shape.length() != 4 {
    abort("flatten2d 仅支持 4D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  let h = shape[2]
  let w = shape[3]
  let out = tensor_reshape(vx, [n, c * h * w])
  self.record_op(
    out,
    Flatten2d,
    [x],
    [tensor_shape_to_tensor(shape)],
    [],
    false,
  )
}

///|
pub fn ComputationGraph::reshape(
  self : ComputationGraph,
  x : Int,
  shape : Array[Int],
) -> Int {
  let vx = self.nodes[x].value
  let out = tensor_reshape(vx, shape)
  self.record_op(
    out,
    Reshape,
    [x],
    [tensor_shape_to_tensor(vx.get_shape())],
    [],
    false,
  )
}

///|
pub fn ComputationGraph::transpose(
  self : ComputationGraph,
  x : Int,
  axes : Array[Int],
) -> Int {
  let vx = self.nodes[x].value
  let out = tensor_transpose(vx, axes)
  self.record_op(out, Transpose, [x], [tensor_shape_to_tensor(axes)], [], false)
}

///|
pub fn ComputationGraph::log_softmax(
  self : ComputationGraph,
  x : Int,
  axis : Int,
) -> Int {
  let vx = self.nodes[x].value
  let lse = tensor_logsumexp_axis(vx, axis, keepdim=true)
  let out = tensor_sub(vx, lse)
  let probs = tensor_exp(out)
  self.record_op(out, LogSoftmax, [x], [probs], [axis.to_double()], false)
}

///|
pub fn ComputationGraph::logsumexp(
  self : ComputationGraph,
  x : Int,
  axis : Int,
  keepdim? : Bool = false,
) -> Int {
  let vx = self.nodes[x].value
  let out = tensor_logsumexp_axis(vx, axis, keepdim~)
  self.record_op(out, LogSumExp, [x], [], [axis.to_double()], keepdim)
}

///|
pub fn ComputationGraph::softmax_cross_entropy(
  self : ComputationGraph,
  logits : Int,
  labels : Array[Int],
) -> Int {
  let v = self.nodes[logits].value
  let (out, probs) = tensor_softmax_cross_entropy(v, labels)
  self.record_op(
    out,
    SoftmaxCrossEntropy,
    [logits],
    [probs, labels_to_tensor(labels)],
    [],
    false,
  )
}

///|
pub fn ComputationGraph::nll_loss(
  self : ComputationGraph,
  log_probs : Int,
  labels : Array[Int],
) -> Int {
  let v = self.nodes[log_probs].value
  let shape = v.get_shape()
  if shape.length() != 2 {
    abort("nll_loss 仅支持 2D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  if labels.length() != n {
    abort("标签长度不匹配")
  }
  let mut loss = 0.0
  for i = 0; i < n; i = i + 1 {
    let cls = labels[i]
    if cls < 0 || cls >= c {
      abort("类别索引越界")
    }
    loss = loss - v.get([i, cls])
  }
  let loss_val = loss / n.to_double()
  let out = @math.Tensor::new([loss_val], [1])
  self.record_op(
    out,
    NLLLoss,
    [log_probs],
    [labels_to_tensor(labels)],
    [],
    false,
  )
}

///|
pub fn ComputationGraph::mse_loss(
  self : ComputationGraph,
  pred : Int,
  target : Int,
) -> Int {
  let vp = self.nodes[pred].value
  let vt = self.nodes[target].value
  let diff = tensor_sub(vp, vt)
  let shape = diff.get_shape()
  let strides = compute_strides(shape)
  let total = diff.total_size()
  let mut sum = 0.0
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    let v = diff.get(idx)
    sum = sum + v * v
  }
  let loss_val = sum / total.to_double()
  let out = @math.Tensor::new([loss_val], [1])
  self.record_op(out, MSELoss, [pred, target], [], [], false)
}

///|
pub fn ComputationGraph::mae_loss(
  self : ComputationGraph,
  pred : Int,
  target : Int,
) -> Int {
  let vp = self.nodes[pred].value
  let vt = self.nodes[target].value
  let diff = tensor_sub(vp, vt)
  let shape = diff.get_shape()
  let strides = compute_strides(shape)
  let total = diff.total_size()
  let mut sum = 0.0
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    sum = sum + diff.get(idx).abs()
  }
  let loss_val = sum / total.to_double()
  let out = @math.Tensor::new([loss_val], [1])
  self.record_op(out, MAELoss, [pred, target], [], [], false)
}

///|
fn topo_collect(
  g : ComputationGraph,
  root : Int,
  visited : Array[Bool],
  order : Array[Int],
) -> Unit {
  if visited[root] {
    return
  }
  visited[root] = true
  let parents = g.nodes[root].parents
  for i = 0; i < parents.length(); i = i + 1 {
    topo_collect(g, parents[i], visited, order)
  }
  order.push(root)
}

///|
pub fn ComputationGraph::backward(self : ComputationGraph, root : Int) -> Unit {
  self.backward_with_grad(root, tensor_ones_like(self.nodes[root].value))
}

///|
pub fn ComputationGraph::backward_with_grad(
  self : ComputationGraph,
  root : Int,
  grad_root : @math.Tensor,
) -> Unit {
  if root < 0 || root >= self.nodes.length() {
    abort("root 超出范围")
  }
  // 清零梯度
  for i = 0; i < self.nodes.length(); i = i + 1 {
    self.nodes[i].grad = tensor_zeros_like(self.nodes[i].value)
  }
  self.nodes[root].grad = grad_root
  let visited = Array::make(self.nodes.length(), false)
  let order = []
  topo_collect(self, root, visited, order)
  // 反向传播
  for idx = order.length() - 1; idx >= 0; idx = idx - 1 {
    let node_id = order[idx]
    let node = self.nodes[node_id]
    if not(node.requires_grad) {
      continue
    }
    match node.op {
      Input => ()
      Add => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let shape0 = self.nodes[p0].value.get_shape()
        let shape1 = self.nodes[p1].value.get_shape()
        let grad0 = tensor_reduce_to_shape(g, shape0)
        let grad1 = tensor_reduce_to_shape(g, shape1)
        tensor_add_inplace(self.nodes[p0].grad, grad0)
        tensor_add_inplace(self.nodes[p1].grad, grad1)
      }
      Sub => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let shape0 = self.nodes[p0].value.get_shape()
        let shape1 = self.nodes[p1].value.get_shape()
        let grad0 = tensor_reduce_to_shape(g, shape0)
        let grad1 = tensor_reduce_to_shape(tensor_neg(g), shape1)
        tensor_add_inplace(self.nodes[p0].grad, grad0)
        tensor_add_inplace(self.nodes[p1].grad, grad1)
      }
      Mul => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let v0 = self.nodes[p0].value
        let v1 = self.nodes[p1].value
        let grad0_full = tensor_mul(g, v1)
        let grad1_full = tensor_mul(g, v0)
        let shape0 = v0.get_shape()
        let shape1 = v1.get_shape()
        let grad0 = tensor_reduce_to_shape(grad0_full, shape0)
        let grad1 = tensor_reduce_to_shape(grad1_full, shape1)
        tensor_add_inplace(self.nodes[p0].grad, grad0)
        tensor_add_inplace(self.nodes[p1].grad, grad1)
      }
      Relu => {
        let g = node.grad
        let p = node.parents[0]
        let mask = tensor_mask_relu(self.nodes[p].value)
        let back = tensor_hadamard(g, mask)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      MatMul => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let a = self.nodes[p0].value
        let b = self.nodes[p1].value
        let shape_a = a.get_shape()
        let shape_b = b.get_shape()
        let m = shape_a[0]
        let k = shape_a[1]
        let n = shape_b[1]
        let grad_a = @math.Tensor::zeros([m, k])
        let grad_b = @math.Tensor::zeros([k, n])
        for i = 0; i < m; i = i + 1 {
          for j = 0; j < n; j = j + 1 {
            let gij = g.get([i, j])
            for p = 0; p < k; p = p + 1 {
              grad_a.set([i, p], grad_a.get([i, p]) + gij * b.get([p, j]))
              grad_b.set([p, j], grad_b.get([p, j]) + a.get([i, p]) * gij)
            }
          }
        }
        tensor_add_inplace(self.nodes[p0].grad, grad_a)
        tensor_add_inplace(self.nodes[p1].grad, grad_b)
      }
      AddBias => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        tensor_add_inplace(self.nodes[p0].grad, g)
        let shape_g = g.get_shape()
        let n = shape_g[0]
        let m = shape_g[1]
        let grad_b = @math.Tensor::zeros([m])
        for i = 0; i < n; i = i + 1 {
          for j = 0; j < m; j = j + 1 {
            grad_b.set([j], grad_b.get([j]) + g.get([i, j]))
          }
        }
        tensor_add_inplace(self.nodes[p1].grad, grad_b)
      }
      Sum => {
        let g = node.grad
        let p = node.parents[0]
        let scalar = g.get([0])
        let base = tensor_ones_like(self.nodes[p].value)
        let back = tensor_scale(base, scalar)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      SumAxis => {
        let g = node.grad
        let p = node.parents[0]
        let shape = tensor_tensor_to_shape(node.aux_tensors[0])
        let axis = node.aux_scalars[0].to_int()
        let back = tensor_expand_for_axis(g, shape, axis, node.aux_bool)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      MeanAxis => {
        let g = node.grad
        let p = node.parents[0]
        let shape = tensor_tensor_to_shape(node.aux_tensors[0])
        let axis = node.aux_scalars[0].to_int()
        let back = tensor_expand_for_axis(g, shape, axis, node.aux_bool)
        let scale = 1.0 / shape[axis].to_double()
        tensor_add_inplace(self.nodes[p].grad, tensor_scale(back, scale))
      }
      Dropout => {
        let g = node.grad
        let p = node.parents[0]
        let mask = node.aux_tensors[0]
        let back = tensor_hadamard(g, mask)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      Dropout2d => {
        let g = node.grad
        let p = node.parents[0]
        let mask = node.aux_tensors[0]
        let back = tensor_hadamard(g, mask)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      BatchNorm1d => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let p2 = node.parents[2]
        let x = self.nodes[p0].value
        let gamma = self.nodes[p1].value
        let shape = x.get_shape()
        if shape.length() != 2 {
          abort("batchnorm1d 仅支持 2D 张量")
        }
        let n = shape[0]
        let m = shape[1]
        let mean = node.aux_tensors[0]
        let variance = node.aux_tensors[1]
        let eps = node.aux_scalars[0]
        let grad_x = @math.Tensor::zeros([n, m])
        let grad_gamma = @math.Tensor::zeros([m])
        let grad_beta = @math.Tensor::zeros([m])
        for j = 0; j < m; j = j + 1 {
          let mean_j = mean.get([j])
          let var_j = variance.get([j])
          let inv_std = 1.0 / (var_j + eps).sqrt()
          if node.aux_bool {
            let mut sum_dxhat = 0.0
            let mut sum_dxhat_xmu = 0.0
            let mut sum_xmu = 0.0
            let mut sum_dy = 0.0
            let mut sum_dy_xhat = 0.0
            for i = 0; i < n; i = i + 1 {
              let xmu = x.get([i, j]) - mean_j
              sum_xmu = sum_xmu + xmu
              let dy = g.get([i, j])
              let xhat = xmu * inv_std
              sum_dy = sum_dy + dy
              sum_dy_xhat = sum_dy_xhat + dy * xhat
              let dxhat = dy * gamma.get([j])
              sum_dxhat = sum_dxhat + dxhat
              sum_dxhat_xmu = sum_dxhat_xmu + dxhat * xmu
            }
            grad_gamma.set([j], grad_gamma.get([j]) + sum_dy_xhat)
            grad_beta.set([j], grad_beta.get([j]) + sum_dy)
            let inv_std3 = inv_std * inv_std * inv_std
            let dvar = sum_dxhat_xmu * -0.5 * inv_std3
            let dmu = sum_dxhat * -inv_std +
              dvar * -2.0 * sum_xmu / n.to_double()
            for i = 0; i < n; i = i + 1 {
              let xmu = x.get([i, j]) - mean_j
              let dxhat = g.get([i, j]) * gamma.get([j])
              let dx = dxhat * inv_std +
                dvar * 2.0 * xmu / n.to_double() +
                dmu / n.to_double()
              grad_x.set([i, j], grad_x.get([i, j]) + dx)
            }
          } else {
            let mut sum_dy = 0.0
            let mut sum_dy_xhat = 0.0
            for i = 0; i < n; i = i + 1 {
              let dy = g.get([i, j])
              let xhat = (x.get([i, j]) - mean_j) * inv_std
              sum_dy = sum_dy + dy
              sum_dy_xhat = sum_dy_xhat + dy * xhat
              let dx = dy * gamma.get([j]) * inv_std
              grad_x.set([i, j], grad_x.get([i, j]) + dx)
            }
            grad_gamma.set([j], grad_gamma.get([j]) + sum_dy_xhat)
            grad_beta.set([j], grad_beta.get([j]) + sum_dy)
          }
        }
        tensor_add_inplace(self.nodes[p0].grad, grad_x)
        tensor_add_inplace(self.nodes[p1].grad, grad_gamma)
        tensor_add_inplace(self.nodes[p2].grad, grad_beta)
      }
      BatchNorm2d => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let p2 = node.parents[2]
        let x = self.nodes[p0].value
        let gamma = self.nodes[p1].value
        let shape = x.get_shape()
        if shape.length() != 4 {
          abort("batchnorm2d 仅支持 4D 张量")
        }
        let n = shape[0]
        let c = shape[1]
        let h = shape[2]
        let w = shape[3]
        let mean = node.aux_tensors[0]
        let variance = node.aux_tensors[1]
        let eps = node.aux_scalars[0]
        let grad_x = @math.Tensor::zeros([n, c, h, w])
        let grad_gamma = @math.Tensor::zeros([c])
        let grad_beta = @math.Tensor::zeros([c])
        let nhw = (n * h * w).to_double()
        for ci = 0; ci < c; ci = ci + 1 {
          let mean_c = mean.get([ci])
          let var_c = variance.get([ci])
          let inv_std = 1.0 / (var_c + eps).sqrt()
          let mut sum_dxhat = 0.0
          let mut sum_dxhat_xmu = 0.0
          let mut sum_xmu = 0.0
          let mut sum_dy = 0.0
          let mut sum_dy_xhat = 0.0
          for ni = 0; ni < n; ni = ni + 1 {
            for hi = 0; hi < h; hi = hi + 1 {
              for wi = 0; wi < w; wi = wi + 1 {
                let xmu = x.get([ni, ci, hi, wi]) - mean_c
                sum_xmu = sum_xmu + xmu
                let dy = g.get([ni, ci, hi, wi])
                let xhat = xmu * inv_std
                sum_dy = sum_dy + dy
                sum_dy_xhat = sum_dy_xhat + dy * xhat
                let dxhat = dy * gamma.get([ci])
                sum_dxhat = sum_dxhat + dxhat
                sum_dxhat_xmu = sum_dxhat_xmu + dxhat * xmu
              }
            }
          }
          grad_gamma.set([ci], grad_gamma.get([ci]) + sum_dy_xhat)
          grad_beta.set([ci], grad_beta.get([ci]) + sum_dy)
          let inv_std3 = inv_std * inv_std * inv_std
          let dvar = sum_dxhat_xmu * -0.5 * inv_std3
          let dmu = sum_dxhat * -inv_std + dvar * -2.0 * sum_xmu / nhw
          for ni = 0; ni < n; ni = ni + 1 {
            for hi = 0; hi < h; hi = hi + 1 {
              for wi = 0; wi < w; wi = wi + 1 {
                let xmu = x.get([ni, ci, hi, wi]) - mean_c
                let dxhat = g.get([ni, ci, hi, wi]) * gamma.get([ci])
                let dx = dxhat * inv_std + dvar * 2.0 * xmu / nhw + dmu / nhw
                grad_x.set([ni, ci, hi, wi], grad_x.get([ni, ci, hi, wi]) + dx)
              }
            }
          }
        }
        tensor_add_inplace(self.nodes[p0].grad, grad_x)
        tensor_add_inplace(self.nodes[p1].grad, grad_gamma)
        tensor_add_inplace(self.nodes[p2].grad, grad_beta)
      }
      LayerNorm => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let p2 = node.parents[2]
        let x = self.nodes[p0].value
        let gamma = self.nodes[p1].value
        let shape = x.get_shape()
        if shape.length() != 2 {
          abort("layernorm 仅支持 2D 张量")
        }
        let n = shape[0]
        let m = shape[1]
        let mean = node.aux_tensors[0]
        let variance = node.aux_tensors[1]
        let eps = node.aux_scalars[0]
        let grad_x = @math.Tensor::zeros([n, m])
        let grad_gamma = @math.Tensor::zeros([m])
        let grad_beta = @math.Tensor::zeros([m])
        for i = 0; i < n; i = i + 1 {
          let mean_i = mean.get([i])
          let var_i = variance.get([i])
          let inv_std = 1.0 / (var_i + eps).sqrt()
          let mut sum_dxhat = 0.0
          let mut sum_dxhat_xmu = 0.0
          let mut sum_xmu = 0.0
          let mut sum_dy = 0.0
          let mut sum_dy_xhat = 0.0
          for j = 0; j < m; j = j + 1 {
            let xmu = x.get([i, j]) - mean_i
            sum_xmu = sum_xmu + xmu
            let dy = g.get([i, j])
            let xhat = xmu * inv_std
            sum_dy = sum_dy + dy
            sum_dy_xhat = sum_dy_xhat + dy * xhat
            let dxhat = dy * gamma.get([j])
            sum_dxhat = sum_dxhat + dxhat
            sum_dxhat_xmu = sum_dxhat_xmu + dxhat * xmu
          }
          let inv_std3 = inv_std * inv_std * inv_std
          let dvar = sum_dxhat_xmu * -0.5 * inv_std3
          let dmu = sum_dxhat * -inv_std + dvar * -2.0 * sum_xmu / m.to_double()
          for j = 0; j < m; j = j + 1 {
            let xmu = x.get([i, j]) - mean_i
            let dxhat = g.get([i, j]) * gamma.get([j])
            let dx = dxhat * inv_std +
              dvar * 2.0 * xmu / m.to_double() +
              dmu / m.to_double()
            grad_x.set([i, j], grad_x.get([i, j]) + dx)
          }
          for j = 0; j < m; j = j + 1 {
            let xhat = (x.get([i, j]) - mean_i) * inv_std
            grad_gamma.set([j], grad_gamma.get([j]) + g.get([i, j]) * xhat)
            grad_beta.set([j], grad_beta.get([j]) + g.get([i, j]))
          }
        }
        tensor_add_inplace(self.nodes[p0].grad, grad_x)
        tensor_add_inplace(self.nodes[p1].grad, grad_gamma)
        tensor_add_inplace(self.nodes[p2].grad, grad_beta)
      }
      Embedding => {
        let g = node.grad
        let weight_id = node.parents[0]
        let w = self.nodes[weight_id].value
        let shape_w = w.get_shape()
        let indices = node.aux_tensors[0]
        let n = indices.total_size()
        let dim = shape_w[1]
        let grad_w = @math.Tensor::zeros([shape_w[0], dim])
        for i = 0; i < n; i = i + 1 {
          let row = indices.get([i]).to_int()
          for j = 0; j < dim; j = j + 1 {
            grad_w.set([row, j], grad_w.get([row, j]) + g.get([i, j]))
          }
        }
        tensor_add_inplace(self.nodes[weight_id].grad, grad_w)
      }
      Conv2d => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let p2 = node.parents[2]
        let x = self.nodes[p0].value
        let w = self.nodes[p1].value
        let shape_x = x.get_shape()
        let shape_w = w.get_shape()
        let n = shape_x[0]
        let cin = shape_x[1]
        let h = shape_x[2]
        let wi = shape_x[3]
        let cout = shape_w[0]
        let kh = shape_w[2]
        let kw = shape_w[3]
        let shape_g = g.get_shape()
        let out_h = shape_g[2]
        let out_w = shape_g[3]
        let stride = node.aux_scalars[0].to_int()
        let padding = node.aux_scalars[1].to_int()
        let grad_x = @math.Tensor::zeros([n, cin, h, wi])
        let grad_w = @math.Tensor::zeros([cout, cin, kh, kw])
        let grad_b = @math.Tensor::zeros([cout])
        for ni = 0; ni < n; ni = ni + 1 {
          for co = 0; co < cout; co = co + 1 {
            for oh = 0; oh < out_h; oh = oh + 1 {
              for ow = 0; ow < out_w; ow = ow + 1 {
                let gg = g.get([ni, co, oh, ow])
                grad_b.set([co], grad_b.get([co]) + gg)
                for ci = 0; ci < cin; ci = ci + 1 {
                  for khi = 0; khi < kh; khi = khi + 1 {
                    for kwi = 0; kwi < kw; kwi = kwi + 1 {
                      let ih = oh * stride + khi - padding
                      let iw = ow * stride + kwi - padding
                      if ih >= 0 && ih < h && iw >= 0 && iw < wi {
                        grad_x.set(
                          [ni, ci, ih, iw],
                          grad_x.get([ni, ci, ih, iw]) +
                          gg * w.get([co, ci, khi, kwi]),
                        )
                        grad_w.set(
                          [co, ci, khi, kwi],
                          grad_w.get([co, ci, khi, kwi]) +
                          gg * x.get([ni, ci, ih, iw]),
                        )
                      }
                    }
                  }
                }
              }
            }
          }
        }
        tensor_add_inplace(self.nodes[p0].grad, grad_x)
        tensor_add_inplace(self.nodes[p1].grad, grad_w)
        tensor_add_inplace(self.nodes[p2].grad, grad_b)
      }
      MaxPool2d => {
        let g = node.grad
        let p = node.parents[0]
        let x = self.nodes[p].value
        let shape = x.get_shape()
        let n = shape[0]
        let c = shape[1]
        let out_shape = g.get_shape()
        let out_h = out_shape[2]
        let out_w = out_shape[3]
        let idx_h = node.aux_tensors[0]
        let idx_w = node.aux_tensors[1]
        let grad_x = @math.Tensor::zeros(shape)
        for ni = 0; ni < n; ni = ni + 1 {
          for ci = 0; ci < c; ci = ci + 1 {
            for oh = 0; oh < out_h; oh = oh + 1 {
              for ow = 0; ow < out_w; ow = ow + 1 {
                let ih = idx_h.get([ni, ci, oh, ow]).to_int()
                let iw = idx_w.get([ni, ci, oh, ow]).to_int()
                grad_x.set(
                  [ni, ci, ih, iw],
                  grad_x.get([ni, ci, ih, iw]) + g.get([ni, ci, oh, ow]),
                )
              }
            }
          }
        }
        tensor_add_inplace(self.nodes[p].grad, grad_x)
      }
      AvgPool2d => {
        let g = node.grad
        let p = node.parents[0]
        let x = self.nodes[p].value
        let shape = x.get_shape()
        let n = shape[0]
        let c = shape[1]
        let h = shape[2]
        let w = shape[3]
        let out_shape = g.get_shape()
        let out_h = out_shape[2]
        let out_w = out_shape[3]
        let kernel = node.aux_scalars[0].to_int()
        let stride = node.aux_scalars[1].to_int()
        let grad_x = @math.Tensor::zeros([n, c, h, w])
        let scale = 1.0 / (kernel * kernel).to_double()
        for ni = 0; ni < n; ni = ni + 1 {
          for ci = 0; ci < c; ci = ci + 1 {
            for oh = 0; oh < out_h; oh = oh + 1 {
              for ow = 0; ow < out_w; ow = ow + 1 {
                let gg = g.get([ni, ci, oh, ow]) * scale
                for khi = 0; khi < kernel; khi = khi + 1 {
                  for kwi = 0; kwi < kernel; kwi = kwi + 1 {
                    let ih = oh * stride + khi
                    let iw = ow * stride + kwi
                    grad_x.set(
                      [ni, ci, ih, iw],
                      grad_x.get([ni, ci, ih, iw]) + gg,
                    )
                  }
                }
              }
            }
          }
        }
        tensor_add_inplace(self.nodes[p].grad, grad_x)
      }
      Flatten2d => {
        let g = node.grad
        let p = node.parents[0]
        let shape_t = node.aux_tensors[0]
        let shape = tensor_tensor_to_shape(shape_t)
        let back = tensor_reshape(g, shape)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      SoftmaxCrossEntropy => {
        let g = node.grad
        let p = node.parents[0]
        let probs = node.aux_tensors[0]
        let labels = node.aux_tensors[1]
        let shape = probs.get_shape()
        let n = shape[0]
        let c = shape[1]
        let grad_logits = @math.Tensor::zeros([n, c])
        let scale = g.get([0]) / n.to_double()
        for i = 0; i < n; i = i + 1 {
          let cls = labels.get([i]).to_int()
          for j = 0; j < c; j = j + 1 {
            let y = if j == cls { 1.0 } else { 0.0 }
            let v = (probs.get([i, j]) - y) * scale
            grad_logits.set([i, j], v)
          }
        }
        tensor_add_inplace(self.nodes[p].grad, grad_logits)
      }
      NLLLoss => {
        let g = node.grad
        let p = node.parents[0]
        let labels = node.aux_tensors[0]
        let shape = self.nodes[p].value.get_shape()
        let n = shape[0]
        let c = shape[1]
        let grad_logp = @math.Tensor::zeros([n, c])
        let scale = -g.get([0]) / n.to_double()
        for i = 0; i < n; i = i + 1 {
          let cls = labels.get([i]).to_int()
          grad_logp.set([i, cls], scale)
        }
        tensor_add_inplace(self.nodes[p].grad, grad_logp)
      }
      MSELoss => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let v0 = self.nodes[p0].value
        let v1 = self.nodes[p1].value
        let diff = tensor_sub(v0, v1)
        let total = diff.total_size()
        let scale = 2.0 * g.get([0]) / total.to_double()
        let grad_full = tensor_scale(diff, scale)
        let grad0 = tensor_reduce_to_shape(grad_full, v0.get_shape())
        let grad1 = tensor_reduce_to_shape(
          tensor_neg(grad_full),
          v1.get_shape(),
        )
        tensor_add_inplace(self.nodes[p0].grad, grad0)
        tensor_add_inplace(self.nodes[p1].grad, grad1)
      }
      MAELoss => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let v0 = self.nodes[p0].value
        let v1 = self.nodes[p1].value
        let diff = tensor_sub(v0, v1)
        let total = diff.total_size()
        let scale = g.get([0]) / total.to_double()
        let sign = tensor_sign(diff)
        let grad_full = tensor_scale(sign, scale)
        let grad0 = tensor_reduce_to_shape(grad_full, v0.get_shape())
        let grad1 = tensor_reduce_to_shape(
          tensor_neg(grad_full),
          v1.get_shape(),
        )
        tensor_add_inplace(self.nodes[p0].grad, grad0)
        tensor_add_inplace(self.nodes[p1].grad, grad1)
      }
      Reshape => {
        let g = node.grad
        let p = node.parents[0]
        let shape = tensor_tensor_to_shape(node.aux_tensors[0])
        let back = tensor_reshape(g, shape)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      Transpose => {
        let g = node.grad
        let p = node.parents[0]
        let x = self.nodes[p].value
        let shape = x.get_shape()
        let axes = tensor_tensor_to_shape(node.aux_tensors[0])
        let real_axes = if axes.length() == 0 {
          let rev = Array::make(shape.length(), 0)
          for i = 0; i < shape.length(); i = i + 1 {
            rev[i] = shape.length() - 1 - i
          }
          rev
        } else {
          axes
        }
        if real_axes.length() != shape.length() {
          abort("transpose 轴长度不匹配")
        }
        let inv = Array::make(real_axes.length(), 0)
        for i = 0; i < real_axes.length(); i = i + 1 {
          let ax = real_axes[i]
          inv[ax] = i
        }
        let back = tensor_transpose(g, inv)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      LogSoftmax => {
        let g = node.grad
        let p = node.parents[0]
        let probs = node.aux_tensors[0]
        let axis = node.aux_scalars[0].to_int()
        let sum_g = tensor_sum_axes(g, [axis], keepdim=true)
        let back = tensor_sub(g, tensor_mul(probs, sum_g))
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      LogSumExp => {
        let g = node.grad
        let p = node.parents[0]
        let x = self.nodes[p].value
        let axis = node.aux_scalars[0].to_int()
        let keepdim = node.aux_bool
        let lse = node.value
        let lse_b = tensor_expand_for_axis(lse, x.get_shape(), axis, keepdim)
        let weights = tensor_exp(tensor_sub(x, lse_b))
        let g_b = tensor_expand_for_axis(g, x.get_shape(), axis, keepdim)
        let back = tensor_mul(weights, g_b)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
    }
  }
}

///|
pub fn ComputationGraph::grad_of(
  self : ComputationGraph,
  id : Int,
) -> @math.Tensor {
  self.nodes[id].grad
}

///|
pub fn ComputationGraph::value_of(
  self : ComputationGraph,
  id : Int,
) -> @math.Tensor {
  self.nodes[id].value
}
