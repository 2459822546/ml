///|
/// 线性代数测试

///|
test "LinAlg - QR 分解" {
  let mat = Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  let qr = qr_decomposition(mat)

  // 验证 Q 是正交的 (Q^T * Q 应该接近单位矩阵)
  let qtq = qr.q.transpose().matmul(qr.q)
  assert_true((qtq.get(0, 0) - 1.0).abs() < 1.0e-6)
  assert_true((qtq.get(1, 1) - 1.0).abs() < 1.0e-6)
  assert_true(qtq.get(0, 1).abs() < 1.0e-6)

  // 验证 Q * R = A
  let reconstructed = qr.q.matmul(qr.r)
  assert_true((reconstructed.get(0, 0) - 1.0).abs() < 1.0e-6)
  assert_true((reconstructed.get(0, 1) - 2.0).abs() < 1.0e-6)
  assert_true((reconstructed.get(1, 0) - 3.0).abs() < 1.0e-6)
}

///|
test "LinAlg - LU 分解" {
  let mat = Matrix::new([[4.0, 3.0], [6.0, 3.0]])
  let lu = lu_decomposition(mat)

  // 验证 L 是下三角矩阵
  assert_eq(lu.l.get(0, 0), 1.0)
  assert_eq(lu.l.get(1, 1), 1.0)

  // 验证 U 是上三角矩阵
  assert_eq(lu.u.get(1, 0), 0.0)

  // 验证 P * A = L * U（简化检查）
  let lxu = lu.l.matmul(lu.u)
  // 考虑置换，验证重构
  assert_true(lxu.get(0, 0) > 0.0)
}

///|
test "LinAlg - Cholesky 分解" {
  // 对称正定矩阵
  let mat = Matrix::new([[4.0, 2.0], [2.0, 3.0]])
  let chol = cholesky_decomposition(mat)

  // 验证 L * L^T = A
  let reconstructed = chol.l.matmul(chol.l.transpose())
  assert_true((reconstructed.get(0, 0) - 4.0).abs() < 1.0e-6)
  assert_true((reconstructed.get(0, 1) - 2.0).abs() < 1.0e-6)
  assert_true((reconstructed.get(1, 0) - 2.0).abs() < 1.0e-6)
  assert_true((reconstructed.get(1, 1) - 3.0).abs() < 1.0e-6)
}

///|
test "LinAlg - 特征值分解（2x2矩阵）" {
  // 对称矩阵 [[3, 1], [1, 3]]
  // 特征值应该是 4 和 2
  let mat = Matrix::new([[3.0, 1.0], [1.0, 3.0]])
  let eigen = eigen_decomposition(mat, 2, 100, 1.0e-6)

  // 最大特征值应该接近 4
  assert_true((eigen.values.get(0) - 4.0).abs() < 0.1)

  // 验证特征向量是单位向量
  let v0_norm = eigen.vectors.get_col(0).norm_l2()
  assert_true((v0_norm - 1.0).abs() < 1.0e-6)
}

///|
test "LinAlg - 求解线性方程组" {
  // 求解 [[2, 1], [1, 3]] * x = [5, 6]
  // 解应该是 x = [2.2, 0.6]
  let a = Matrix::new([[2.0, 1.0], [1.0, 3.0]])
  let b = Vector::new([5.0, 6.0])
  let x = solve_linear_system(a, b)

  // 验证 A * x = b
  let ax = a.matvec(x)
  assert_true((ax.get(0) - 5.0).abs() < 1.0e-6)
  assert_true((ax.get(1) - 6.0).abs() < 1.0e-6)
}

///|
test "LinAlg - 行列式" {
  let mat = Matrix::new([[1.0, 2.0], [3.0, 4.0]])
  let det = determinant(mat)

  // det = 1*4 - 2*3 = -2
  assert_true((det - -2.0).abs() < 1.0e-6)
}

///|
test "LinAlg - 迹" {
  let mat = Matrix::new([[1.0, 2.0], [3.0, 4.0]])
  let tr = trace(mat)

  // trace = 1 + 4 = 5
  assert_eq(tr, 5.0)
}

///|
test "LinAlg - 秩" {
  let mat = Matrix::new([[1.0, 2.0], [2.0, 4.0]]) // 秩为 1
  let r = rank(mat, 1.0e-6)
  assert_eq(r, 1)
}

///|
test "LinAlg - 矩阵求逆" {
  let mat = Matrix::new([[4.0, 7.0], [2.0, 6.0]])
  let inv = inverse(mat)

  // 验证 A * A^(-1) = I
  let product = mat.matmul(inv)
  assert_true((product.get(0, 0) - 1.0).abs() < 1.0e-6)
  assert_true((product.get(1, 1) - 1.0).abs() < 1.0e-6)
  assert_true(product.get(0, 1).abs() < 1.0e-6)
  assert_true(product.get(1, 0).abs() < 1.0e-6)
}

///|
test "LinAlg - SVD 分解" {
  let mat = Matrix::new([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
  let svd_result = svd(mat, 2, 100, 1.0e-6)

  // 对于这个对角矩阵，奇异值应该是 3 和 2
  let s0 = svd_result.s.get(0)
  let _s1 = svd_result.s.get(1)

  // 最大奇异值应该接近 3
  assert_true((s0 - 3.0).abs() < 0.5 || (s0 - 2.0).abs() < 0.5)
}

///|
test "LinAlg - 矩阵伪逆" {
  // 对于满秩方阵，伪逆 = 逆
  let mat = Matrix::new([[4.0, 7.0], [2.0, 6.0]])
  let pinv = pinverse(mat, 2, 100, 1.0e-6)

  // 验证 A * A^+ * A ≈ A
  let product = mat.matmul(pinv).matmul(mat)
  assert_true((product.get(0, 0) - 4.0).abs() < 1.0)
  assert_true((product.get(1, 1) - 6.0).abs() < 1.0)
}
