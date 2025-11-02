///|
/// Matrix 测试
test "Matrix - 创建和基本属性" {
  let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let (rows, cols) = m.shape()
  assert_eq(rows, 2)
  assert_eq(cols, 3)
  assert_eq(m.get(0, 0), 1.0)
  assert_eq(m.get(1, 2), 6.0)
}

///|
test "Matrix - 单位矩阵" {
  let identity = Matrix::identity(3)
  assert_eq(identity.get(0, 0), 1.0)
  assert_eq(identity.get(1, 1), 1.0)
  assert_eq(identity.get(2, 2), 1.0)
  assert_eq(identity.get(0, 1), 0.0)
  assert_eq(identity.get(1, 0), 0.0)
}

///|
test "Matrix - 矩阵加法" {
  let m1 = Matrix::new([[1.0, 2.0], [3.0, 4.0]])
  let m2 = Matrix::new([[5.0, 6.0], [7.0, 8.0]])
  let m_add = m1.op_add(m2)
  assert_eq(m_add.get(0, 0), 6.0)
  assert_eq(m_add.get(0, 1), 8.0)
  assert_eq(m_add.get(1, 0), 10.0)
  assert_eq(m_add.get(1, 1), 12.0)
}

///|
test "Matrix - 矩阵乘法" {
  let m1 = Matrix::new([[1.0, 2.0], [3.0, 4.0]])
  let m2 = Matrix::new([[5.0, 6.0], [7.0, 8.0]])
  let m_mul = m1.matmul(m2)
  // [1*5+2*7, 1*6+2*8]   [19, 22]
  // [3*5+4*7, 3*6+4*8] = [43, 50]
  assert_eq(m_mul.get(0, 0), 19.0)
  assert_eq(m_mul.get(0, 1), 22.0)
  assert_eq(m_mul.get(1, 0), 43.0)
  assert_eq(m_mul.get(1, 1), 50.0)
}

///|
test "Matrix - 转置" {
  let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let m_t = m.transpose()
  let (rows, cols) = m_t.shape()
  assert_eq(rows, 3)
  assert_eq(cols, 2)
  assert_eq(m_t.get(0, 0), 1.0)
  assert_eq(m_t.get(0, 1), 4.0)
  assert_eq(m_t.get(1, 0), 2.0)
  assert_eq(m_t.get(2, 1), 6.0)
}

///|
test "Matrix - 矩阵向量乘法" {
  let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]])
  let v = Vector::new([5.0, 6.0])
  let result = m.matvec(v)
  // [1*5+2*6, 3*5+4*6] = [17, 39]
  assert_eq(result.get(0), 17.0)
  assert_eq(result.get(1), 39.0)
}

///|
test "Matrix - 统计函数" {
  let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  assert_eq(m.sum(), 21.0)
  assert_eq(m.mean(), 3.5)
  let row_sums = m.sum_rows()
  assert_eq(row_sums.get(0), 6.0)
  assert_eq(row_sums.get(1), 15.0)
  let col_sums = m.sum_cols()
  assert_eq(col_sums.get(0), 5.0)
  assert_eq(col_sums.get(1), 7.0)
  assert_eq(col_sums.get(2), 9.0)
}


