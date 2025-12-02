///| 统计函数模块
/// 以更轻量的风格提供常用统计计算

///|
/// 复制并排序数据（不改原数组）
fn sorted_copy(data : Array[Double]) -> Array[Double] {
  if data.length() == 0 {
    abort("至少需要一个数据点")
  }
  let sorted = Array::make(data.length(), 0.0)
  for i = 0; i < data.length(); i = i + 1 {
    sorted[i] = data[i]
  }
  sorted.sort_by(fn(a, b) { if a < b { -1 } else if a > b { 1 } else { 0 } })
  sorted
}

///|
/// 简单的均值实现，避免重复代码
fn mean(data : Array[Double]) -> Double {
  if data.length() == 0 {
    abort("空数组无法计算均值")
  }
  let mut total = 0.0
  for i = 0; i < data.length(); i = i + 1 {
    total = total + data[i]
  }
  total / data.length().to_double()
}

///|
/// 中位数
pub fn median(data : Array[Double]) -> Double {
  let sorted = sorted_copy(data)
  let n = sorted.length()
  if n % 2 == 1 {
    sorted[n / 2]
  } else {
    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
  }
}

///|
/// 百分位数
pub fn percentile(data : Array[Double], p : Double) -> Double {
  if p < 0.0 || p > 100.0 {
    abort("百分位数必须在 0-100 之间")
  }
  let sorted = sorted_copy(data)
  let n = sorted.length()
  let index = p / 100.0 * (n - 1).to_double()
  let lower = index.to_int()
  let upper = lower + 1
  if upper >= n {
    sorted[n - 1]
  } else {
    let weight = index - lower.to_double()
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
  }
}

///|
/// 四分位距 (IQR)
pub fn iqr(data : Array[Double]) -> Double {
  let q1 = percentile(data, 25.0)
  let q3 = percentile(data, 75.0)
  q3 - q1
}

///|
/// 偏度 (Skewness)
pub fn skewness(data : Array[Double]) -> Double {
  if data.length() < 3 {
    abort("计算偏度至少需要 3 个数据点")
  }
  let n = data.length()
  let mean_val = mean(data)
  let mut sum_sq = 0.0
  let mut sum_cube = 0.0
  for i = 0; i < n; i = i + 1 {
    let diff = data[i] - mean_val
    let diff_sq = diff * diff
    sum_sq = sum_sq + diff_sq
    sum_cube = sum_cube + diff_sq * diff
  }
  let variance = sum_sq / n.to_double()
  let std = variance.sqrt()
  if std == 0.0 {
    return 0.0
  }
  sum_cube / n.to_double() / (std * std * std)
}

///|
/// 峰度 (Kurtosis)
pub fn kurtosis(data : Array[Double]) -> Double {
  if data.length() < 4 {
    abort("计算峰度至少需要 4 个数据点")
  }
  let n = data.length()
  let mean_val = mean(data)
  let mut sum_sq = 0.0
  let mut sum_fourth = 0.0
  for i = 0; i < n; i = i + 1 {
    let diff = data[i] - mean_val
    let diff_sq = diff * diff
    sum_sq = sum_sq + diff_sq
    sum_fourth = sum_fourth + diff_sq * diff_sq
  }
  let variance = sum_sq / n.to_double()
  if variance == 0.0 {
    return 0.0
  }
  sum_fourth / n.to_double() / (variance * variance) - 3.0
}

///| 相关性分析

///|
/// 计算协方差
pub fn covariance(x : Array[Double], y : Array[Double]) -> Double {
  if x.length() != y.length() {
    abort("数组长度必须相同")
  }
  if x.length() == 0 {
    abort("空数组无法计算协方差")
  }
  let n = x.length()

  // 计算均值
  let mut sum_x = 0.0
  let mut sum_y = 0.0
  for i = 0; i < n; i = i + 1 {
    sum_x = sum_x + x[i]
    sum_y = sum_y + y[i]
  }
  let mean_x = sum_x / n.to_double()
  let mean_y = sum_y / n.to_double()

  // 计算协方差
  let mut cov = 0.0
  for i = 0; i < n; i = i + 1 {
    cov = cov + (x[i] - mean_x) * (y[i] - mean_y)
  }
  cov / n.to_double()
}

///|
/// 计算皮尔逊相关系数
pub fn correlation(x : Array[Double], y : Array[Double]) -> Double {
  if x.length() != y.length() {
    abort("数组长度必须相同")
  }
  if x.length() == 0 {
    abort("空数组无法计算相关系数")
  }
  let n = x.length()

  // 计算均值
  let mut sum_x = 0.0
  let mut sum_y = 0.0
  for i = 0; i < n; i = i + 1 {
    sum_x = sum_x + x[i]
    sum_y = sum_y + y[i]
  }
  let mean_x = sum_x / n.to_double()
  let mean_y = sum_y / n.to_double()

  // 计算相关系数
  let mut numerator = 0.0
  let mut sum_sq_x = 0.0
  let mut sum_sq_y = 0.0
  for i = 0; i < n; i = i + 1 {
    let diff_x = x[i] - mean_x
    let diff_y = y[i] - mean_y
    numerator = numerator + diff_x * diff_y
    sum_sq_x = sum_sq_x + diff_x * diff_x
    sum_sq_y = sum_sq_y + diff_y * diff_y
  }
  let denominator = (sum_sq_x * sum_sq_y).sqrt()
  if denominator == 0.0 {
    return 0.0
  }
  numerator / denominator
}

///| 距离度量

///|
/// 欧几里得距离
pub fn euclidean_distance(x : Array[Double], y : Array[Double]) -> Double {
  if x.length() != y.length() {
    abort("数组长度必须相同")
  }
  let mut sum_sq = 0.0
  for i = 0; i < x.length(); i = i + 1 {
    let diff = x[i] - y[i]
    sum_sq = sum_sq + diff * diff
  }
  sum_sq.sqrt()
}

///|
/// 曼哈顿距离
pub fn manhattan_distance(x : Array[Double], y : Array[Double]) -> Double {
  if x.length() != y.length() {
    abort("数组长度必须相同")
  }
  let mut sum = 0.0
  for i = 0; i < x.length(); i = i + 1 {
    sum = sum + (x[i] - y[i]).abs()
  }
  sum
}

///|
/// 余弦相似度
pub fn cosine_similarity(x : Array[Double], y : Array[Double]) -> Double {
  if x.length() != y.length() {
    abort("数组长度必须相同")
  }
  let mut dot_product = 0.0
  let mut norm_x = 0.0
  let mut norm_y = 0.0
  for i = 0; i < x.length(); i = i + 1 {
    dot_product = dot_product + x[i] * y[i]
    norm_x = norm_x + x[i] * x[i]
    norm_y = norm_y + y[i] * y[i]
  }
  let denominator = norm_x.sqrt() * norm_y.sqrt()
  if denominator == 0.0 {
    return 0.0
  }
  dot_product / denominator
}
