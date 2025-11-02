///|
/// 线性代数 (Linear Algebra)
/// 包含矩阵分解、特征值计算等高级线性代数运算

///|
/// QR 分解结果
pub struct QRDecomposition {
  q : Matrix // 正交矩阵
  r : Matrix // 上三角矩阵
}

///|
/// SVD 分解结果
pub struct SVDResult {
  u : Matrix // 左奇异向量
  s : Vector // 奇异值
  vt : Matrix // 右奇异向量的转置
}

///|
/// 特征值分解结果
pub struct EigenDecomposition {
  values : Vector // 特征值
  vectors : Matrix // 特征向量（列向量）
}

///|
/// LU 分解结果
pub struct LUDecomposition {
  l : Matrix // 下三角矩阵
  u : Matrix // 上三角矩阵
  p : Array[Int] // 置换数组
}

///|
/// Cholesky 分解结果（仅用于对称正定矩阵）
pub struct CholeskyDecomposition {
  l : Matrix // 下三角矩阵
}

///|
/// QR 分解 (使用 Gram-Schmidt 正交化)
pub fn qr_decomposition(mat : Matrix) -> QRDecomposition {
  let (m, n) = mat.shape()
  if m < n {
    abort("QR分解要求行数 >= 列数")
  }

  // 初始化 Q 和 R
  let q_data = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    q_data[i] = Array::make(m, 0.0)
  }
  let r_data = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    r_data[i] = Array::make(n, 0.0)
  }

  // Gram-Schmidt 正交化
  for j = 0; j < n; j = j + 1 {
    // 获取第 j 列
    let mut col_j = mat.get_col(j)

    // 减去之前所有正交向量的投影
    for i = 0; i < j; i = i + 1 {
      let q_i = Vector::new(q_data[i])
      let r_ij = q_i.dot(col_j)
      r_data[i][j] = r_ij

      // col_j = col_j - r_ij * q_i
      col_j = col_j.op_sub(q_i.op_mul(r_ij))
    }

    // 归一化
    let norm = col_j.norm_l2()
    r_data[j][j] = norm
    if norm > 1.0e-10 {
      col_j = col_j.op_div(norm)
    }

    // 存储 Q 的第 j 列
    for k = 0; k < m; k = k + 1 {
      q_data[j][k] = col_j.get(k)
    }
  }

  // 转置 q_data 以得到正确的 Q 矩阵
  let q_final = Array::make(m, [])
  for i = 0; i < m; i = i + 1 {
    let row = Array::make(n, 0.0)
    for j = 0; j < n; j = j + 1 {
      row[j] = q_data[j][i]
    }
    q_final[i] = row
  }
  { q: Matrix::new(q_final), r: Matrix::new(r_data) }
}

///|
/// 幂迭代法求最大特征值和对应的特征向量
fn power_iteration(mat : Matrix, max_iter : Int) -> (Double, Vector) {
  let (n, _) = mat.shape()

  // 初始化随机向量
  let mut v = Vector::ones(n)
  v = v.normalize()
  let mut eigenvalue = 0.0
  for iter = 0; iter < max_iter; iter = iter + 1 {
    // v_new = A * v
    let v_new = mat.matvec(v)

    // 计算特征值 (Rayleigh quotient)
    let new_eigenvalue = v.dot(v_new)

    // 归一化
    v = v_new.normalize()

    // 检查收敛
    if (new_eigenvalue - eigenvalue).abs() < 1.0e-10 {
      eigenvalue = new_eigenvalue
      break
    }
    eigenvalue = new_eigenvalue
  }
  (eigenvalue, v)
}

///|
/// 计算最大的 k 个特征值和特征向量（使用幂迭代法）
/// 注意：这是一个简化实现，适用于对称矩阵
pub fn eigen_decomposition(
  mat : Matrix,
  k : Int,
  max_iter : Int,
  _tol : Double,
) -> EigenDecomposition {
  let (n, m) = mat.shape()
  if n != m {
    abort("特征值分解要求方阵")
  }
  let eigenvalues = Array::make(k, 0.0)
  let eigenvectors_data = Array::make(k, [])
  let current_mat = mat.copy()
  for i = 0; i < k; i = i + 1 {
    // 使用幂迭代法找到最大特征值和特征向量
    let (eigenvalue, eigenvector) = power_iteration(current_mat, max_iter)
    eigenvalues[i] = eigenvalue
    eigenvectors_data[i] = eigenvector.to_array()

    // Deflation: 移除这个特征值的贡献
    // A' = A - λ * v * v^T
    for row = 0; row < n; row = row + 1 {
      for col = 0; col < n; col = col + 1 {
        let old_val = current_mat.get(row, col)
        let deflation = eigenvalue * eigenvector.get(row) * eigenvector.get(col)
        current_mat.set(row, col, old_val - deflation)
      }
    }
  }

  // 转置特征向量数据以创建矩阵
  let vectors_matrix_data = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    let row = Array::make(k, 0.0)
    for j = 0; j < k; j = j + 1 {
      if i < eigenvectors_data[j].length() {
        row[j] = eigenvectors_data[j][i]
      }
    }
    vectors_matrix_data[i] = row
  }
  {
    values: Vector::new(eigenvalues),
    vectors: Matrix::new(vectors_matrix_data),
  }
}

///|
/// SVD 分解（使用迭代方法的简化实现）
/// 计算 A = U * Σ * V^T
pub fn svd(mat : Matrix, k : Int, max_iter : Int, tol : Double) -> SVDResult {
  let (m, _) = mat.shape()

  // 计算 A^T * A 的特征值分解以获得 V
  let ata = mat.transpose().matmul(mat)
  let eigen = eigen_decomposition(ata, k, max_iter, tol)

  // 奇异值是特征值的平方根
  let singular_values = eigen.values.map(fn(x) { x.sqrt() })

  // V 就是 A^T * A 的特征向量
  let v = eigen.vectors
  let vt = v.transpose()

  // 计算 U = A * V * Σ^(-1)
  let u_data = Array::make(m, [])
  for i = 0; i < m; i = i + 1 {
    let row = Array::make(k, 0.0)
    u_data[i] = row
  }
  for j = 0; j < k; j = j + 1 {
    let v_col = v.get_col(j)
    let av = mat.matvec(v_col)
    let sigma_j = singular_values.get(j)
    if sigma_j > 1.0e-10 {
      let u_col = av.op_div(sigma_j)
      for i = 0; i < m; i = i + 1 {
        u_data[i][j] = u_col.get(i)
      }
    }
  }
  { u: Matrix::new(u_data), s: singular_values, vt }
}

///|
/// LU 分解（带部分选主元）
pub fn lu_decomposition(mat : Matrix) -> LUDecomposition {
  let (n, m) = mat.shape()
  if n != m {
    abort("LU分解要求方阵")
  }

  // 复制矩阵
  let a = mat.copy()
  let l_data = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    let row = Array::make(n, 0.0)
    for j = 0; j < n; j = j + 1 {
      if i == j {
        row[j] = 1.0
      }
    }
    l_data[i] = row
  }
  let perm = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    perm[i] = i
  }

  // Doolittle 算法
  for k = 0; k < n; k = k + 1 {
    // 部分选主元
    let mut max_val = a.get(k, k).abs()
    let mut max_idx = k
    for i = k + 1; i < n; i = i + 1 {
      let val = a.get(i, k).abs()
      if val > max_val {
        max_val = val
        max_idx = i
      }
    }

    // 交换行
    if max_idx != k {
      for j = 0; j < n; j = j + 1 {
        let temp = a.get(k, j)
        a.set(k, j, a.get(max_idx, j))
        a.set(max_idx, j, temp)
        if j < k {
          let temp_l = l_data[k][j]
          l_data[k][j] = l_data[max_idx][j]
          l_data[max_idx][j] = temp_l
        }
      }
      let temp_p = perm[k]
      perm[k] = perm[max_idx]
      perm[max_idx] = temp_p
    }

    // 计算 L 和 U
    for i = k + 1; i < n; i = i + 1 {
      let factor = a.get(i, k) / a.get(k, k)
      l_data[i][k] = factor
      for j = k; j < n; j = j + 1 {
        let val = a.get(i, j) - factor * a.get(k, j)
        a.set(i, j, val)
      }
    }
  }
  { l: Matrix::new(l_data), u: a, p: perm }
}

///|
/// Cholesky 分解（仅用于对称正定矩阵）
/// 分解 A = L * L^T
pub fn cholesky_decomposition(mat : Matrix) -> CholeskyDecomposition {
  let (n, m) = mat.shape()
  if n != m {
    abort("Cholesky分解要求方阵")
  }
  let l_data = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    l_data[i] = Array::make(n, 0.0)
  }
  for i = 0; i < n; i = i + 1 {
    for j = 0; j <= i; j = j + 1 {
      let mut sum = 0.0
      if i == j {
        // 对角元素
        for k = 0; k < j; k = k + 1 {
          sum = sum + l_data[j][k] * l_data[j][k]
        }
        let val = mat.get(j, j) - sum
        if val <= 0.0 {
          abort("矩阵不是正定的")
        }
        l_data[j][j] = val.sqrt()
      } else {
        // 非对角元素
        for k = 0; k < j; k = k + 1 {
          sum = sum + l_data[i][k] * l_data[j][k]
        }
        l_data[i][j] = (mat.get(i, j) - sum) / l_data[j][j]
      }
    }
  }
  { l: Matrix::new(l_data) }
}

///|
/// 求解线性方程组 Ax = b (使用 LU 分解)
pub fn solve_linear_system(a : Matrix, b : Vector) -> Vector {
  let (n, m) = a.shape()
  if n != m {
    abort("系数矩阵必须是方阵")
  }
  if b.size() != n {
    abort("向量维度不匹配")
  }
  let lu = lu_decomposition(a)

  // 应用置换到 b
  let b_perm = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    b_perm[i] = b.get(lu.p[i])
  }
  let y = Vector::new(b_perm)

  // 前向替换 L * y = P * b
  for i = 0; i < n; i = i + 1 {
    let mut sum = 0.0
    for j = 0; j < i; j = j + 1 {
      sum = sum + lu.l.get(i, j) * y.get(j)
    }
    y.set(i, y.get(i) - sum)
  }

  // 后向替换 U * x = y
  let x = Vector::zeros(n)
  for i_rev = 0; i_rev < n; i_rev = i_rev + 1 {
    let i = n - 1 - i_rev
    let mut sum = 0.0
    for j = i + 1; j < n; j = j + 1 {
      sum = sum + lu.u.get(i, j) * x.get(j)
    }
    x.set(i, (y.get(i) - sum) / lu.u.get(i, i))
  }
  x
}

///|
/// 计算矩阵的行列式（使用 LU 分解）
pub fn determinant(mat : Matrix) -> Double {
  let (n, m) = mat.shape()
  if n != m {
    abort("行列式要求方阵")
  }
  let lu = lu_decomposition(mat)

  // 行列式 = U 对角线元素的乘积 * 置换符号
  let mut det = 1.0
  for i = 0; i < n; i = i + 1 {
    det = det * lu.u.get(i, i)
  }

  // 计算置换符号
  let mut swaps = 0
  let perm_copy = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    perm_copy[i] = lu.p[i]
  }
  for i = 0; i < n; i = i + 1 {
    while perm_copy[i] != i {
      let j = perm_copy[i]
      let temp = perm_copy[i]
      perm_copy[i] = perm_copy[j]
      perm_copy[j] = temp
      swaps = swaps + 1
    }
  }
  if swaps % 2 == 1 {
    det = -det
  }
  det
}

///|
/// 计算矩阵的迹（对角线元素之和）
pub fn trace(mat : Matrix) -> Double {
  let (n, m) = mat.shape()
  if n != m {
    abort("迹要求方阵")
  }
  let mut tr = 0.0
  for i = 0; i < n; i = i + 1 {
    tr = tr + mat.get(i, i)
  }
  tr
}

///|
/// 计算矩阵的秩（使用 QR 分解）
pub fn rank(mat : Matrix, tol : Double) -> Int {
  let qr = qr_decomposition(mat)
  let (_, n) = qr.r.shape()
  let mut r = 0
  for i = 0; i < n; i = i + 1 {
    if qr.r.get(i, i).abs() > tol {
      r = r + 1
    }
  }
  r
}

///|
/// 矩阵求逆（使用 LU 分解）
pub fn inverse(mat : Matrix) -> Matrix {
  let (n, m) = mat.shape()
  if n != m {
    abort("求逆要求方阵")
  }
  let inv_data = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    inv_data[i] = Array::make(n, 0.0)
  }

  // 对每一列单位向量求解
  for j = 0; j < n; j = j + 1 {
    let e = Array::make(n, 0.0)
    e[j] = 1.0
    let col = solve_linear_system(mat, Vector::new(e))
    for i = 0; i < n; i = i + 1 {
      inv_data[i][j] = col.get(i)
    }
  }
  Matrix::new(inv_data)
}

///|
/// 矩阵的伪逆（Moore-Penrose 伪逆，使用 SVD）
pub fn pinverse(mat : Matrix, k : Int, max_iter : Int, tol : Double) -> Matrix {
  let svd_result = svd(mat, k, max_iter, tol)

  // A^+ = V * Σ^+ * U^T
  // Σ^+ 是 Σ 的伪逆（非零奇异值取倒数）

  let sigma_pinv_data = Array::make(k, [])
  for i = 0; i < k; i = i + 1 {
    let row = Array::make(k, 0.0)
    let s_i = svd_result.s.get(i)
    if s_i > tol {
      row[i] = 1.0 / s_i
    }
    sigma_pinv_data[i] = row
  }
  let sigma_pinv = Matrix::new(sigma_pinv_data)

  // V * Σ^+ * U^T
  svd_result.vt.transpose().matmul(sigma_pinv).matmul(svd_result.u.transpose())
}
