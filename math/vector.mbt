///|
/// 向量 (Vector)
/// 一维数值数组，支持各种向量运算
pub struct Vector {
  data : Array[Double]
}

///|
/// 构造函数
pub fn Vector::new(data : Array[Double]) -> Vector {
  { data, }
}

///|
pub fn Vector::from_list(list : Array[Double]) -> Vector {
  Vector::new(list)
}

///|
pub fn Vector::zeros(size : Int) -> Vector {
  Vector::new(Array::make(size, 0.0))
}

///|
pub fn Vector::ones(size : Int) -> Vector {
  Vector::new(Array::make(size, 1.0))
}

///|
pub fn Vector::fill(size : Int, value : Double) -> Vector {
  Vector::new(Array::make(size, value))
}

///|
/// 基本属性
pub fn size(self : Vector) -> Int {
  self.data.length()
}

///|
pub fn get(self : Vector, index : Int) -> Double {
  self.data[index]
}

///|
pub fn set(self : Vector, index : Int, value : Double) -> Unit {
  self.data[index] = value
}

///|
pub fn to_array(self : Vector) -> Array[Double] {
  self.data
}

///| 向量运算

///|
/// 向量加法
pub fn op_add(self : Vector, other : Vector) -> Vector {
  if self.size() != other.size() {
    abort("向量维度不匹配")
  }
  let result = Array::make(self.size(), 0.0)
  for i = 0; i < self.size(); i = i + 1 {
    result[i] = self.data[i] + other.data[i]
  }
  Vector::new(result)
}

///|
/// 向量减法
pub fn op_sub(self : Vector, other : Vector) -> Vector {
  if self.size() != other.size() {
    abort("向量维度不匹配")
  }
  let result = Array::make(self.size(), 0.0)
  for i = 0; i < self.size(); i = i + 1 {
    result[i] = self.data[i] - other.data[i]
  }
  Vector::new(result)
}

///|
/// 标量乘法
pub fn op_mul(self : Vector, scalar : Double) -> Vector {
  let result = Array::make(self.size(), 0.0)
  for i = 0; i < self.size(); i = i + 1 {
    result[i] = self.data[i] * scalar
  }
  Vector::new(result)
}

///|
/// 标量除法
pub fn op_div(self : Vector, scalar : Double) -> Vector {
  if scalar == 0.0 {
    abort("除数不能为零")
  }
  let result = Array::make(self.size(), 0.0)
  for i = 0; i < self.size(); i = i + 1 {
    result[i] = self.data[i] / scalar
  }
  Vector::new(result)
}

///|
/// 元素级乘法 (Hadamard product)
pub fn hadamard(self : Vector, other : Vector) -> Vector {
  if self.size() != other.size() {
    abort("向量维度不匹配")
  }
  let result = Array::make(self.size(), 0.0)
  for i = 0; i < self.size(); i = i + 1 {
    result[i] = self.data[i] * other.data[i]
  }
  Vector::new(result)
}

///|
/// 点积 (内积)
pub fn dot(self : Vector, other : Vector) -> Double {
  if self.size() != other.size() {
    abort("向量维度不匹配")
  }
  let mut sum = 0.0
  for i = 0; i < self.size(); i = i + 1 {
    sum = sum + self.data[i] * other.data[i]
  }
  sum
}

///| 范数计算

///|
/// L1 范数 (曼哈顿距离)
pub fn norm_l1(self : Vector) -> Double {
  let mut sum = 0.0
  for i = 0; i < self.size(); i = i + 1 {
    sum = sum + self.data[i].abs()
  }
  sum
}

///|
/// L2 范数 (欧几里得范数)
pub fn norm_l2(self : Vector) -> Double {
  self.dot(self).sqrt()
}

///|
/// 无穷范数 (最大绝对值)
pub fn norm_inf(self : Vector) -> Double {
  let mut max_val = 0.0
  for i = 0; i < self.size(); i = i + 1 {
    let abs_val = self.data[i].abs()
    if abs_val > max_val {
      max_val = abs_val
    }
  }
  max_val
}

///|
/// 归一化向量 (L2)
pub fn normalize(self : Vector) -> Vector {
  let norm = self.norm_l2()
  if norm == 0.0 {
    return self
  }
  self.op_div(norm)
}

///| 统计函数

///|
/// 求和
pub fn sum(self : Vector) -> Double {
  let mut total = 0.0
  for i = 0; i < self.size(); i = i + 1 {
    total = total + self.data[i]
  }
  total
}

///|
/// 均值
pub fn mean(self : Vector) -> Double {
  if self.size() == 0 {
    abort("空向量没有均值")
  }
  self.sum() / self.size().to_double()
}

///|
/// 方差
pub fn variance(self : Vector) -> Double {
  if self.size() == 0 {
    abort("空向量没有方差")
  }
  let mean_val = self.mean()
  let mut sum_sq = 0.0
  for i = 0; i < self.size(); i = i + 1 {
    let diff = self.data[i] - mean_val
    sum_sq = sum_sq + diff * diff
  }
  sum_sq / self.size().to_double()
}

///|
/// 标准差
pub fn std(self : Vector) -> Double {
  self.variance().sqrt()
}

///|
/// 最小值
pub fn min(self : Vector) -> Double {
  if self.size() == 0 {
    abort("空向量没有最小值")
  }
  let mut min_val = self.data[0]
  for i = 1; i < self.size(); i = i + 1 {
    if self.data[i] < min_val {
      min_val = self.data[i]
    }
  }
  min_val
}

///|
/// 最大值
pub fn max(self : Vector) -> Double {
  if self.size() == 0 {
    abort("空向量没有最大值")
  }
  let mut max_val = self.data[0]
  for i = 1; i < self.size(); i = i + 1 {
    if self.data[i] > max_val {
      max_val = self.data[i]
    }
  }
  max_val
}

///| 实用函数

///|
/// 复制向量
pub fn copy(self : Vector) -> Vector {
  let result = Array::make(self.size(), 0.0)
  for i = 0; i < self.size(); i = i + 1 {
    result[i] = self.data[i]
  }
  Vector::new(result)
}

///|
/// 应用函数到每个元素
pub fn map(self : Vector, f : (Double) -> Double) -> Vector {
  let result = Array::make(self.size(), 0.0)
  for i = 0; i < self.size(); i = i + 1 {
    result[i] = f(self.data[i])
  }
  Vector::new(result)
}

///|
/// 打印向量
pub fn Vector::to_string(self : Vector) -> String {
  let mut str = "Vector["
  for i = 0; i < self.size(); i = i + 1 {
    str = str + self.data[i].to_string()
    if i < self.size() - 1 {
      str = str + ", "
    }
  }
  str + "]"
}


