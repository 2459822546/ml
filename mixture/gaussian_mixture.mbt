///|
/// 对角协方差的高斯混合模型（EM）

///|
fn log_gaussian_diag(
  x : @math.Vector,
  mean : @math.Vector,
  variance : @math.Vector,
) -> Double {
  let dim = mean.size()
  let mut total = 0.0
  for i = 0; i < dim; i = i + 1 {
    let v = variance.get(i)
    let safe_var = if v <= 1.0e-9 { 1.0e-9 } else { v }
    let diff = x.get(i) - mean.get(i)
    let log_term = @moonbitlang/core/math.ln(6.283185307179586 * safe_var)
    total = total - 0.5 * (diff * diff / safe_var + log_term)
  }
  total
}

///|
fn init_covariance(x : @math.Matrix) -> @math.Vector {
  let (n_samples, n_features) = x.shape()
  let mean = Array::make(n_features, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    for f = 0; f < n_features; f = f + 1 {
      mean[f] = mean[f] + row.get(f)
    }
  }
  for f = 0; f < n_features; f = f + 1 {
    mean[f] = mean[f] / n_samples.to_double()
  }
  let variance = Array::make(n_features, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    for f = 0; f < n_features; f = f + 1 {
      let diff = row.get(f) - mean[f]
      variance[f] = variance[f] + diff * diff
    }
  }
  for f = 0; f < n_features; f = f + 1 {
    variance[f] = variance[f] / n_samples.to_double() + 1.0e-6
  }
  @math.Vector::new(variance)
}

///|
/// 高斯混合模型（对角协方差）
pub struct GaussianMixture {
  n_components : Int
  max_iter : Int
  tol : Double
  mut weights : Array[Double]
  mut means : Array[@math.Vector]
  mut covariances : Array[@math.Vector]
  mut is_fitted : Bool
}

///|
pub fn GaussianMixture::new(
  n_components : Int,
  max_iter? : Int = 50,
  tol? : Double = 1.0e-3,
) -> GaussianMixture {
  if n_components <= 0 {
    abort("n_components 必须大于 0")
  }
  if max_iter <= 0 {
    abort("max_iter 必须大于 0")
  }
  if tol <= 0.0 {
    abort("tol 必须大于 0")
  }
  {
    n_components,
    max_iter,
    tol,
    weights: [],
    means: [],
    covariances: [],
    is_fitted: false,
  }
}

///|
/// 拟合模型（EM）
pub fn GaussianMixture::fit(self : GaussianMixture, x : @math.Matrix) -> Unit {
  let (n_samples, n_features) = x.shape()
  if n_samples == 0 {
    abort("数据不能为空")
  }
  if self.n_components > n_samples {
    abort("n_components 不能大于样本数")
  }
  self.weights = Array::make(
    self.n_components,
    1.0 / self.n_components.to_double(),
  )
  self.means = Array::make(self.n_components, @math.Vector::zeros(n_features))
  self.covariances = Array::make(
    self.n_components,
    @math.Vector::fill(n_features, 1.0),
  )
  let base_cov = init_covariance(x)
  for k = 0; k < self.n_components; k = k + 1 {
    let row = x.get_row(k)
    self.means[k] = row
    self.covariances[k] = base_cov
  }
  let resp = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    resp[i] = Array::make(self.n_components, 0.0)
  }
  let mut prev_log_likelihood = -@double.infinity
  for _iter = 0; _iter < self.max_iter; _iter = _iter + 1 {
    // E step
    let mut log_likelihood = 0.0
    for i = 0; i < n_samples; i = i + 1 {
      let row = x.get_row(i)
      let tmp = Array::make(self.n_components, 0.0)
      let mut max_log = -@double.infinity
      for k = 0; k < self.n_components; k = k + 1 {
        let lp = @moonbitlang/core/math.ln(self.weights[k]) +
          log_gaussian_diag(row, self.means[k], self.covariances[k])
        tmp[k] = lp
        if lp > max_log {
          max_log = lp
        }
      }
      let mut sum_exp = 0.0
      for k = 0; k < self.n_components; k = k + 1 {
        sum_exp = sum_exp + @moonbitlang/core/math.exp(tmp[k] - max_log)
      }
      let log_sum = max_log + @moonbitlang/core/math.ln(sum_exp)
      for k = 0; k < self.n_components; k = k + 1 {
        resp[i][k] = @moonbitlang/core/math.exp(tmp[k] - log_sum)
      }
      log_likelihood = log_likelihood + log_sum
    }

    // M step
    let nk = Array::make(self.n_components, 0.0)
    for k = 0; k < self.n_components; k = k + 1 {
      let mut total = 0.0
      for i = 0; i < n_samples; i = i + 1 {
        total = total + resp[i][k]
      }
      nk[k] = total
    }
    for k = 0; k < self.n_components; k = k + 1 {
      let weight = nk[k]
      self.weights[k] = weight / n_samples.to_double()
      let mean_arr = Array::make(n_features, 0.0)
      for i = 0; i < n_samples; i = i + 1 {
        let row = x.get_row(i)
        for f = 0; f < n_features; f = f + 1 {
          mean_arr[f] = mean_arr[f] + resp[i][k] * row.get(f)
        }
      }
      if weight > 0.0 {
        for f = 0; f < n_features; f = f + 1 {
          mean_arr[f] = mean_arr[f] / weight
        }
      }
      self.means[k] = @math.Vector::new(mean_arr)
      let cov_arr = Array::make(n_features, 0.0)
      for i = 0; i < n_samples; i = i + 1 {
        let row = x.get_row(i)
        for f = 0; f < n_features; f = f + 1 {
          let diff = row.get(f) - mean_arr[f]
          cov_arr[f] = cov_arr[f] + resp[i][k] * diff * diff
        }
      }
      if weight > 0.0 {
        for f = 0; f < n_features; f = f + 1 {
          cov_arr[f] = cov_arr[f] / weight + 1.0e-6
        }
      }
      self.covariances[k] = @math.Vector::new(cov_arr)
    }
    if (log_likelihood - prev_log_likelihood).abs() < self.tol {
      break
    }
    prev_log_likelihood = log_likelihood
  }
  self.is_fitted = true
}

///|
/// 预测各成分的责任概率
pub fn GaussianMixture::predict_proba(
  self : GaussianMixture,
  x : @math.Matrix,
) -> @math.Matrix {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let out = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    out[i] = Array::make(self.n_components, 0.0)
  }
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    let tmp = Array::make(self.n_components, 0.0)
    let mut max_log = -@double.infinity
    for k = 0; k < self.n_components; k = k + 1 {
      let lp = @moonbitlang/core/math.ln(self.weights[k]) +
        log_gaussian_diag(row, self.means[k], self.covariances[k])
      tmp[k] = lp
      if lp > max_log {
        max_log = lp
      }
    }
    let mut sum_exp = 0.0
    for k = 0; k < self.n_components; k = k + 1 {
      sum_exp = sum_exp + @moonbitlang/core/math.exp(tmp[k] - max_log)
    }
    let log_sum = max_log + @moonbitlang/core/math.ln(sum_exp)
    for k = 0; k < self.n_components; k = k + 1 {
      out[i][k] = @moonbitlang/core/math.exp(tmp[k] - log_sum)
    }
  }
  @math.Matrix::new(out)
}

///|
/// 预测所属的成分索引
pub fn GaussianMixture::predict(
  self : GaussianMixture,
  x : @math.Matrix,
) -> Array[Int] {
  let resp = self.predict_proba(x)
  let (n_samples, _) = resp.shape()
  let labels = Array::make(n_samples, 0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = resp.get_row(i)
    let mut best = 0
    let mut best_val = row.get(0)
    for k = 1; k < self.n_components; k = k + 1 {
      if row.get(k) > best_val {
        best_val = row.get(k)
        best = k
      }
    }
    labels[i] = best
  }
  labels
}
