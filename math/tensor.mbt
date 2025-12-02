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
  if not(self.shapes_equal(other)) {
    abort("张量形状不匹配")
  }
  let result_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    result_data[i] = self.data[i] + other.data[i]
  }
  Tensor::new(result_data, self.shape)
}

///|
/// 张量减法
pub fn Tensor::op_sub(self : Tensor, other : Tensor) -> Tensor {
  if not(self.shapes_equal(other)) {
    abort("张量形状不匹配")
  }
  let result_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    result_data[i] = self.data[i] - other.data[i]
  }
  Tensor::new(result_data, self.shape)
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
  if not(self.shapes_equal(other)) {
    abort("张量形状不匹配")
  }
  let result_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    result_data[i] = self.data[i] * other.data[i]
  }
  Tensor::new(result_data, self.shape)
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

///| 形状操作

///|
/// 重塑张量
pub fn Tensor::reshape(self : Tensor, new_shape : Array[Int]) -> Tensor {
  let mut new_size = 1
  for i = 0; i < new_shape.length(); i = i + 1 {
    new_size = new_size * new_shape[i]
  }
  if new_size != self.size {
    abort("新形状的总大小必须与原张量相同")
  }

  // 复制数据
  let new_data = Array::make(self.size, 0.0)
  for i = 0; i < self.size; i = i + 1 {
    new_data[i] = self.data[i]
  }
  Tensor::new(new_data, new_shape)
}

///|
/// 转置（只支持2D）
pub fn Tensor::transpose(self : Tensor) -> Tensor {
  if self.ndim() != 2 {
    abort("转置只支持2D张量")
  }
  let rows = self.shape[0]
  let cols = self.shape[1]
  let new_data = Array::make(self.size, 0.0)
  for i = 0; i < rows; i = i + 1 {
    for j = 0; j < cols; j = j + 1 {
      let old_idx = i * cols + j
      let new_idx = j * rows + i
      new_data[new_idx] = self.data[old_idx]
    }
  }
  Tensor::new(new_data, [cols, rows])
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
  if axis < 0 || axis >= self.ndim() {
    abort("轴索引越界")
  }

  // 计算结果形状（移除指定轴）
  let new_shape_list = []
  for i = 0; i < self.shape.length(); i = i + 1 {
    if i != axis {
      new_shape_list.push(self.shape[i])
    }
  }
  let new_shape = Array::make(new_shape_list.length(), 0)
  for i = 0; i < new_shape_list.length(); i = i + 1 {
    new_shape[i] = new_shape_list[i]
  }
  let mut result_size = 1
  for i = 0; i < new_shape.length(); i = i + 1 {
    result_size = result_size * new_shape[i]
  }
  let result_data = Array::make(result_size, 0.0)

  // 对于2D张量的简化实现
  if self.ndim() == 2 {
    let rows = self.shape[0]
    let cols = self.shape[1]
    if axis == 0 {
      // 按行求和，结果是列向量
      for j = 0; j < cols; j = j + 1 {
        let mut sum = 0.0
        for i = 0; i < rows; i = i + 1 {
          sum = sum + self.data[i * cols + j]
        }
        result_data[j] = sum
      }
    } else {
      // 按列求和，结果是行向量
      for i = 0; i < rows; i = i + 1 {
        let mut sum = 0.0
        for j = 0; j < cols; j = j + 1 {
          sum = sum + self.data[i * cols + j]
        }
        result_data[i] = sum
      }
    }
  }
  if new_shape.length() == 0 {
    Tensor::new(result_data, [1])
  } else {
    Tensor::new(result_data, new_shape)
  }
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
