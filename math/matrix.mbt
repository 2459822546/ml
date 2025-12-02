///|
/// 矩阵 (Matrix)
/// 二维数值数组，支持各种矩阵运算
pub struct Matrix {
  data : Array[Array[Double]]
  rows : Int
  cols : Int
}

///|
/// 构造函数
pub fn Matrix::new(data : Array[Array[Double]]) -> Matrix {
  if data.length() == 0 {
    abort("矩阵不能为空")
  }
  let rows = data.length()
  let cols = data[0].length()

  // 验证所有行的列数相同
  for i = 1; i < rows; i = i + 1 {
    if data[i].length() != cols {
      abort("矩阵的所有行必须有相同的列数")
    }
  }
  { data, rows, cols }
}

///|
pub fn Matrix::zeros(rows : Int, cols : Int) -> Matrix {
  let data = Array::make(rows, [])
  for i = 0; i < rows; i = i + 1 {
    data[i] = Array::make(cols, 0.0)
  }
  Matrix::new(data)
}

///|
pub fn Matrix::ones(rows : Int, cols : Int) -> Matrix {
  let data = Array::make(rows, [])
  for i = 0; i < rows; i = i + 1 {
    data[i] = Array::make(cols, 1.0)
  }
  Matrix::new(data)
}

///|
pub fn Matrix::identity(n : Int) -> Matrix {
  let data = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    let row = Array::make(n, 0.0)
    row[i] = 1.0
    data[i] = row
  }
  Matrix::new(data)
}

///|
pub fn Matrix::fill(rows : Int, cols : Int, value : Double) -> Matrix {
  let data = Array::make(rows, [])
  for i = 0; i < rows; i = i + 1 {
    data[i] = Array::make(cols, value)
  }
  Matrix::new(data)
}

///|
/// 基本属性
pub fn Matrix::shape(self : Matrix) -> (Int, Int) {
  (self.rows, self.cols)
}

///|
pub fn Matrix::get(self : Matrix, row : Int, col : Int) -> Double {
  self.data[row][col]
}

///|
pub fn Matrix::set(self : Matrix, row : Int, col : Int, value : Double) -> Unit {
  self.data[row][col] = value
}

///|
pub fn Matrix::get_row(self : Matrix, row : Int) -> Vector {
  Vector::new(self.data[row])
}

///|
pub fn Matrix::get_col(self : Matrix, col : Int) -> Vector {
  let data = Array::make(self.rows, 0.0)
  for i = 0; i < self.rows; i = i + 1 {
    data[i] = self.data[i][col]
  }
  Vector::new(data)
}

///| 矩阵运算

///|
/// 矩阵加法
pub fn Matrix::op_add(self : Matrix, other : Matrix) -> Matrix {
  if self.rows != other.rows || self.cols != other.cols {
    abort("矩阵维度不匹配")
  }
  let result = Array::make(self.rows, [])
  for i = 0; i < self.rows; i = i + 1 {
    let row = Array::make(self.cols, 0.0)
    for j = 0; j < self.cols; j = j + 1 {
      row[j] = self.data[i][j] + other.data[i][j]
    }
    result[i] = row
  }
  Matrix::new(result)
}

///|
/// 矩阵减法
pub fn Matrix::op_sub(self : Matrix, other : Matrix) -> Matrix {
  if self.rows != other.rows || self.cols != other.cols {
    abort("矩阵维度不匹配")
  }
  let result = Array::make(self.rows, [])
  for i = 0; i < self.rows; i = i + 1 {
    let row = Array::make(self.cols, 0.0)
    for j = 0; j < self.cols; j = j + 1 {
      row[j] = self.data[i][j] - other.data[i][j]
    }
    result[i] = row
  }
  Matrix::new(result)
}

///|
/// 标量乘法
pub fn Matrix::op_mul(self : Matrix, scalar : Double) -> Matrix {
  let result = Array::make(self.rows, [])
  for i = 0; i < self.rows; i = i + 1 {
    let row = Array::make(self.cols, 0.0)
    for j = 0; j < self.cols; j = j + 1 {
      row[j] = self.data[i][j] * scalar
    }
    result[i] = row
  }
  Matrix::new(result)
}

///|
/// 矩阵乘法
pub fn Matrix::matmul(self : Matrix, other : Matrix) -> Matrix {
  if self.cols != other.rows {
    abort(
      "矩阵维度不匹配：(\{self.rows}, \{self.cols}) × (\{other.rows}, \{other.cols})",
    )
  }
  let result = Array::make(self.rows, [])
  for i = 0; i < self.rows; i = i + 1 {
    let row = Array::make(other.cols, 0.0)
    for j = 0; j < other.cols; j = j + 1 {
      let mut sum = 0.0
      for k = 0; k < self.cols; k = k + 1 {
        sum = sum + self.data[i][k] * other.data[k][j]
      }
      row[j] = sum
    }
    result[i] = row
  }
  Matrix::new(result)
}

///|
/// 矩阵-向量乘法
pub fn Matrix::matvec(self : Matrix, vec : Vector) -> Vector {
  if self.cols != vec.size() {
    abort("维度不匹配")
  }
  let result = Array::make(self.rows, 0.0)
  for i = 0; i < self.rows; i = i + 1 {
    let mut sum = 0.0
    for j = 0; j < self.cols; j = j + 1 {
      sum = sum + self.data[i][j] * vec.get(j)
    }
    result[i] = sum
  }
  Vector::new(result)
}

///|
/// 元素级乘法 (Hadamard product)
pub fn Matrix::hadamard(self : Matrix, other : Matrix) -> Matrix {
  if self.rows != other.rows || self.cols != other.cols {
    abort("矩阵维度不匹配")
  }
  let result = Array::make(self.rows, [])
  for i = 0; i < self.rows; i = i + 1 {
    let row = Array::make(self.cols, 0.0)
    for j = 0; j < self.cols; j = j + 1 {
      row[j] = self.data[i][j] * other.data[i][j]
    }
    result[i] = row
  }
  Matrix::new(result)
}

///|
/// 转置
pub fn Matrix::transpose(self : Matrix) -> Matrix {
  let result = Array::make(self.cols, [])
  for j = 0; j < self.cols; j = j + 1 {
    let row = Array::make(self.rows, 0.0)
    for i = 0; i < self.rows; i = i + 1 {
      row[i] = self.data[i][j]
    }
    result[j] = row
  }
  Matrix::new(result)
}

///| 统计函数

///|
/// 求和（所有元素）
pub fn Matrix::sum(self : Matrix) -> Double {
  let mut total = 0.0
  for i = 0; i < self.rows; i = i + 1 {
    for j = 0; j < self.cols; j = j + 1 {
      total = total + self.data[i][j]
    }
  }
  total
}

///|
/// 按行求和
pub fn Matrix::sum_rows(self : Matrix) -> Vector {
  let result = Array::make(self.rows, 0.0)
  for i = 0; i < self.rows; i = i + 1 {
    let mut sum = 0.0
    for j = 0; j < self.cols; j = j + 1 {
      sum = sum + self.data[i][j]
    }
    result[i] = sum
  }
  Vector::new(result)
}

///|
/// 按列求和
pub fn Matrix::sum_cols(self : Matrix) -> Vector {
  let result = Array::make(self.cols, 0.0)
  for j = 0; j < self.cols; j = j + 1 {
    let mut sum = 0.0
    for i = 0; i < self.rows; i = i + 1 {
      sum = sum + self.data[i][j]
    }
    result[j] = sum
  }
  Vector::new(result)
}

///|
/// 均值
pub fn Matrix::mean(self : Matrix) -> Double {
  let total = self.rows * self.cols
  if total == 0 {
    abort("空矩阵没有均值")
  }
  self.sum() / total.to_double()
}

///| 范数

///|
/// Frobenius 范数
pub fn Matrix::norm_frobenius(self : Matrix) -> Double {
  let mut sum_sq = 0.0
  for i = 0; i < self.rows; i = i + 1 {
    for j = 0; j < self.cols; j = j + 1 {
      let val = self.data[i][j]
      sum_sq = sum_sq + val * val
    }
  }
  sum_sq.sqrt()
}

///| 实用函数

///|
/// 复制矩阵
pub fn Matrix::copy(self : Matrix) -> Matrix {
  let result = Array::make(self.rows, [])
  for i = 0; i < self.rows; i = i + 1 {
    let row = Array::make(self.cols, 0.0)
    for j = 0; j < self.cols; j = j + 1 {
      row[j] = self.data[i][j]
    }
    result[i] = row
  }
  Matrix::new(result)
}

///|
/// 应用函数到每个元素
pub fn Matrix::map(self : Matrix, f : (Double) -> Double) -> Matrix {
  let result = Array::make(self.rows, [])
  for i = 0; i < self.rows; i = i + 1 {
    let row = Array::make(self.cols, 0.0)
    for j = 0; j < self.cols; j = j + 1 {
      row[j] = f(self.data[i][j])
    }
    result[i] = row
  }
  Matrix::new(result)
}

///|
/// 打印矩阵
pub fn Matrix::to_string(self : Matrix) -> String {
  let mut str = "Matrix[\n"
  for i = 0; i < self.rows; i = i + 1 {
    str = str + "  ["
    for j = 0; j < self.cols; j = j + 1 {
      str = str + self.data[i][j].to_string()
      if j < self.cols - 1 {
        str = str + ", "
      }
    }
    str = str + "]"
    if i < self.rows - 1 {
      str = str + ",\n"
    }
  }
  str + "\n]"
}
