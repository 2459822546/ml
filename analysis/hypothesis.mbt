///|
/// 假设检验与分布相关统计

///|
/// 两独立样本 t 检验（等方差假设），返回 t 统计量
pub fn t_test_independent(x : Array[Double], y : Array[Double]) -> Double {
  if x.length() < 2 || y.length() < 2 {
    abort("每个样本至少需要 2 个值")
  }
  let mean_x = mean_array(x)
  let mean_y = mean_array(y)
  let var_x = variance_array(x)
  let var_y = variance_array(y)
  let nx = x.length().to_double()
  let ny = y.length().to_double()
  let pooled = ((nx - 1.0) * var_x + (ny - 1.0) * var_y) / (nx + ny - 2.0)
  let standard_error = (pooled * (1.0 / nx + 1.0 / ny)).sqrt()
  (mean_x - mean_y) / standard_error
}

///|
/// 卡方拟合优度检验统计量
pub fn chi_square_gof(
  observed : Array[Double],
  expected : Array[Double],
) -> Double {
  if observed.length() != expected.length() {
    abort("observed 与 expected 长度不一致")
  }
  let mut chi2 = 0.0
  for i = 0; i < observed.length(); i = i + 1 {
    let e = expected[i]
    if e == 0.0 {
      abort("期望频次不能为 0")
    }
    let diff = observed[i] - e
    chi2 = chi2 + diff * diff / e
  }
  chi2
}

///|
/// 两样本 Kolmogorov-Smirnov 检验 D 统计量
pub fn ks_test_two_sample(
  sample_a : Array[Double],
  sample_b : Array[Double],
) -> Double {
  if sample_a.length() == 0 || sample_b.length() == 0 {
    abort("样本不能为空")
  }
  let a = Array::make(sample_a.length(), 0.0)
  let b = Array::make(sample_b.length(), 0.0)
  for i = 0; i < sample_a.length(); i = i + 1 {
    a[i] = sample_a[i]
  }
  for i = 0; i < sample_b.length(); i = i + 1 {
    b[i] = sample_b[i]
  }
  a.sort_by(fn(x, y) { if x < y { -1 } else if x > y { 1 } else { 0 } })
  b.sort_by(fn(x, y) { if x < y { -1 } else if x > y { 1 } else { 0 } })
  let mut i = 0
  let mut j = 0
  let na = a.length()
  let nb = b.length()
  let mut d = 0.0
  while i < na && j < nb {
    let v = if a[i] <= b[j] { a[i] } else { b[j] }
    while i < na && a[i] <= v {
      i = i + 1
    }
    while j < nb && b[j] <= v {
      j = j + 1
    }
    let fa = i.to_double() / na.to_double()
    let fb = j.to_double() / nb.to_double()
    let diff = (fa - fb).abs()
    if diff > d {
      d = diff
    }
  }
  d
}

///|
/// Jarque-Bera 正态性检验统计量（简化版，替代 Shapiro-Wilk）
pub fn jarque_bera(data : Array[Double]) -> Double {
  if data.length() < 3 {
    abort("至少需要 3 个样本")
  }
  let n = data.length().to_double()
  let mu = mean_array(data)
  let mut m2 = 0.0
  let mut m3 = 0.0
  let mut m4 = 0.0
  for i = 0; i < data.length(); i = i + 1 {
    let diff = data[i] - mu
    let diff2 = diff * diff
    m2 = m2 + diff2
    m3 = m3 + diff2 * diff
    m4 = m4 + diff2 * diff2
  }
  let s2 = m2 / n
  if s2 == 0.0 {
    return 0.0
  }
  let skew = m3 / n / (s2.sqrt() * s2)
  let kurt = m4 / n / (s2 * s2)
  let excess_kurt = kurt - 3.0
  let jb = n / 6.0 * (skew * skew + 0.25 * excess_kurt * excess_kurt)
  jb
}
