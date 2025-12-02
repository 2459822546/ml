///|
/// 线性回归模型
/// 包括普通最小二乘法、岭回归、Lasso 回归和弹性网络

///|
/// 线性回归 (Ordinary Least Squares)
pub struct LinearRegression {
  mut weights : @math.Vector // 权重系数
  mut intercept : Double // 截距
  mut is_fitted : Bool // 是否已训练
}

///|
/// 岭回归 (Ridge Regression)
pub struct Ridge {
  mut weights : @math.Vector
  mut intercept : Double
  mut is_fitted : Bool
  alpha : Double // L2 正则化系数
}

///|
/// Lasso 回归
pub struct Lasso {
  mut weights : @math.Vector
  mut intercept : Double
  mut is_fitted : Bool
  alpha : Double // L1 正则化系数
  max_iter : Int
  tol : Double
}

///|
/// 弹性网络 (ElasticNet)
pub struct ElasticNet {
  mut weights : @math.Vector
  mut intercept : Double
  mut is_fitted : Bool
  alpha : Double // 正则化强度
  l1_ratio : Double // L1 比例 (0到1之间)
  max_iter : Int
  tol : Double
}

///|
/// ========== 线性回归 ==========

///|
pub fn LinearRegression::new() -> LinearRegression {
  { weights: @math.Vector::zeros(0), intercept: 0.0, is_fitted: false }
}

///|
/// 训练线性回归模型（使用正规方程）
/// X: n_samples x n_features
/// y: n_samples
pub fn LinearRegression::fit(
  self : LinearRegression,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }

  // 添加截距项：X_new = [1, X]
  let x_with_intercept = add_intercept_column(x)

  // 使用正规方程: w = (X^T X)^(-1) X^T y
  let xt = x_with_intercept.transpose()
  let xtx = xt.matmul(x_with_intercept)
  // 为避免 X^T X 奇异，给特征对角线添加极小L2正则
  let eps = 1.0e-8
  for i = 1; i < n_features + 1; i = i + 1 {
    xtx.set(i, i, xtx.get(i, i) + eps)
  }
  let xty = xt.matvec(y)

  // 求解线性方程组
  let w = @math.solve_linear_system(xtx, xty)
  self.intercept = w.get(0)
  let weights_data = Array::make(n_features, 0.0)
  for i = 0; i < n_features; i = i + 1 {
    weights_data[i] = w.get(i + 1)
  }
  self.weights = @math.Vector::new(weights_data)
  self.is_fitted = true
}

///|
/// 预测
pub fn LinearRegression::predict(
  self : LinearRegression,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let predictions = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    predictions[i] = self.weights.dot(row) + self.intercept
  }
  @math.Vector::new(predictions)
}

///|
/// 计算 R² 分数
pub fn LinearRegression::score(
  self : LinearRegression,
  x : @math.Matrix,
  y : @math.Vector,
) -> Double {
  let y_pred = self.predict(x)
  r2_score(y, y_pred)
}

///|
/// ========== 岭回归 ==========

///|
pub fn Ridge::new(alpha : Double) -> Ridge {
  { weights: @math.Vector::zeros(0), intercept: 0.0, is_fitted: false, alpha }
}

///|
/// 训练岭回归模型
pub fn Ridge::fit(self : Ridge, x : @math.Matrix, y : @math.Vector) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }

  // 添加截距项
  let x_with_intercept = add_intercept_column(x)

  // 岭回归正规方程: w = (X^T X + α I)^(-1) X^T y
  let xt = x_with_intercept.transpose()
  let xtx = xt.matmul(x_with_intercept)

  // 添加 L2 正则项（不对截距正则化）
  for i = 1; i < n_features + 1; i = i + 1 {
    let val = xtx.get(i, i)
    xtx.set(i, i, val + self.alpha)
  }
  let xty = xt.matvec(y)
  let w = @math.solve_linear_system(xtx, xty)
  self.intercept = w.get(0)
  let weights_data = Array::make(n_features, 0.0)
  for i = 0; i < n_features; i = i + 1 {
    weights_data[i] = w.get(i + 1)
  }
  self.weights = @math.Vector::new(weights_data)
  self.is_fitted = true
}

///|
pub fn Ridge::predict(self : Ridge, x : @math.Matrix) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let predictions = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    predictions[i] = self.weights.dot(row) + self.intercept
  }
  @math.Vector::new(predictions)
}

///|
pub fn Ridge::score(self : Ridge, x : @math.Matrix, y : @math.Vector) -> Double {
  let y_pred = self.predict(x)
  r2_score(y, y_pred)
}

///|
/// ========== Lasso 回归 ==========

///|
pub fn Lasso::new(alpha : Double, max_iter : Int, tol : Double) -> Lasso {
  {
    weights: @math.Vector::zeros(0),
    intercept: 0.0,
    is_fitted: false,
    alpha,
    max_iter,
    tol,
  }
}

///|
/// 软阈值函数（用于坐标下降）
fn soft_threshold(x : Double, lambda : Double) -> Double {
  if x > lambda {
    x - lambda
  } else if x < -lambda {
    x + lambda
  } else {
    0.0
  }
}

///|
/// 训练 Lasso 回归（使用坐标下降法）
pub fn Lasso::fit(self : Lasso, x : @math.Matrix, y : @math.Vector) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }

  // 标准化 X 和 y
  let x_normalized = normalize_features(x)
  self.intercept = y.mean()

  // 初始化权重
  self.weights = @math.Vector::zeros(n_features)

  // 坐标下降
  for _iter = 0; _iter < self.max_iter; _iter = _iter + 1 {
    let old_weights = self.weights.copy()
    for j = 0; j < n_features; j = j + 1 {
      // 计算残差
      let mut residual = 0.0
      for i = 0; i < n_samples; i = i + 1 {
        let mut pred = self.intercept
        for k = 0; k < n_features; k = k + 1 {
          if k != j {
            pred = pred + x_normalized.get(i, k) * self.weights.get(k)
          }
        }
        residual = residual + x_normalized.get(i, j) * (y.get(i) - pred)
      }

      // 软阈值更新
      let w_j = soft_threshold(residual / n_samples.to_double(), self.alpha)
      self.weights.set(j, w_j)
    }

    // 检查收敛
    let diff = old_weights.op_sub(self.weights).norm_l2()
    if diff < self.tol {
      break
    }
  }
  self.is_fitted = true
}

///|
pub fn Lasso::predict(self : Lasso, x : @math.Matrix) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let x_normalized = normalize_features(x)
  let predictions = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x_normalized.get_row(i)
    predictions[i] = self.weights.dot(row) + self.intercept
  }
  @math.Vector::new(predictions)
}

///|
pub fn Lasso::score(self : Lasso, x : @math.Matrix, y : @math.Vector) -> Double {
  let y_pred = self.predict(x)
  r2_score(y, y_pred)
}

///|
/// ========== 弹性网络 ==========

///|
pub fn ElasticNet::new(
  alpha : Double,
  l1_ratio : Double,
  max_iter : Int,
  tol : Double,
) -> ElasticNet {
  {
    weights: @math.Vector::zeros(0),
    intercept: 0.0,
    is_fitted: false,
    alpha,
    l1_ratio,
    max_iter,
    tol,
  }
}

///|
/// 训练弹性网络（结合 L1 和 L2 正则化）
pub fn ElasticNet::fit(
  self : ElasticNet,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  let x_normalized = normalize_features(x)
  self.intercept = y.mean()
  self.weights = @math.Vector::zeros(n_features)
  let l1_reg = self.alpha * self.l1_ratio
  let l2_reg = self.alpha * (1.0 - self.l1_ratio)

  // 坐标下降
  for _iter = 0; _iter < self.max_iter; _iter = _iter + 1 {
    let old_weights = self.weights.copy()
    for j = 0; j < n_features; j = j + 1 {
      let mut residual = 0.0
      for i = 0; i < n_samples; i = i + 1 {
        let mut pred = self.intercept
        for k = 0; k < n_features; k = k + 1 {
          if k != j {
            pred = pred + x_normalized.get(i, k) * self.weights.get(k)
          }
        }
        residual = residual + x_normalized.get(i, j) * (y.get(i) - pred)
      }

      // 弹性网络更新（结合 L1 和 L2）
      let numerator = soft_threshold(residual / n_samples.to_double(), l1_reg)
      let denominator = 1.0 + l2_reg
      self.weights.set(j, numerator / denominator)
    }

    // 检查收敛
    let diff = old_weights.op_sub(self.weights).norm_l2()
    if diff < self.tol {
      break
    }
  }
  self.is_fitted = true
}

///|
pub fn ElasticNet::predict(self : ElasticNet, x : @math.Matrix) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let x_normalized = normalize_features(x)
  let predictions = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x_normalized.get_row(i)
    predictions[i] = self.weights.dot(row) + self.intercept
  }
  @math.Vector::new(predictions)
}

///|
pub fn ElasticNet::score(
  self : ElasticNet,
  x : @math.Matrix,
  y : @math.Vector,
) -> Double {
  let y_pred = self.predict(x)
  r2_score(y, y_pred)
}

///|
/// ========== 辅助函数 ==========

///|
/// 添加截距列（全为1的列）
fn add_intercept_column(x : @math.Matrix) -> @math.Matrix {
  let (n_samples, n_features) = x.shape()
  let new_data = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(n_features + 1, 0.0)
    row[0] = 1.0
    for j = 0; j < n_features; j = j + 1 {
      row[j + 1] = x.get(i, j)
    }
    new_data[i] = row
  }
  @math.Matrix::new(new_data)
}

///|
/// 标准化特征（均值为0，方差为1）
fn normalize_features(x : @math.Matrix) -> @math.Matrix {
  let (n_samples, n_features) = x.shape()
  let normalized = x.copy()
  for j = 0; j < n_features; j = j + 1 {
    let col = x.get_col(j)
    let mean = col.mean()
    let std = col.std()
    if std > 1.0e-10 {
      for i = 0; i < n_samples; i = i + 1 {
        let val = (x.get(i, j) - mean) / std
        normalized.set(i, j, val)
      }
    }
  }
  normalized
}

///|
/// 计算 R² 分数
fn r2_score(y_true : @math.Vector, y_pred : @math.Vector) -> Double {
  if y_true.size() != y_pred.size() {
    abort("y_true 和 y_pred 长度不匹配")
  }
  let y_mean = y_true.mean()
  let mut ss_tot = 0.0
  let mut ss_res = 0.0
  for i = 0; i < y_true.size(); i = i + 1 {
    let diff = y_true.get(i) - y_mean
    ss_tot = ss_tot + diff * diff
    let residual = y_true.get(i) - y_pred.get(i)
    ss_res = ss_res + residual * residual
  }
  1.0 - ss_res / ss_tot
}
