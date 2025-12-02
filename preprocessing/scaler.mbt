///|
/// 标准化器 (StandardScaler)
/// 将特征标准化为零均值和单位方差
/// 公式: z = (x - mean) / std
pub struct StandardScaler {
  mut mean : Array[Double]
  mut std : Array[Double]
  mut fitted : Bool
}

///|
/// 校验特征矩阵是否为空且各行长度一致
fn validate_feature_matrix(data : Array[Array[Double]]) -> Int {
  if data.length() == 0 {
    abort("数据不能为空")
  }
  let n_features = data[0].length()
  if n_features == 0 {
    abort("特征数量必须大于 0")
  }
  for i = 0; i < data.length(); i = i + 1 {
    if data[i].length() != n_features {
      abort("第 \{i} 行的特征数量与其他行不一致")
    }
  }
  n_features
}

///|
pub fn StandardScaler::new() -> StandardScaler {
  { mean: [], std: [], fitted: false }
}

///|
/// 拟合数据，计算均值和标准差
/// @param data 训练数据，每个元素是一个样本（特征数组）
pub fn StandardScaler::fit(
  self : StandardScaler,
  data : Array[Array[Double]],
) -> Unit {
  let n_features = validate_feature_matrix(data)
  let n_samples = data.length()

  // 初始化均值和标准差数组
  self.mean = Array::make(n_features, 0.0)
  self.std = Array::make(n_features, 0.0)

  // 计算均值
  for i = 0; i < n_features; i = i + 1 {
    let mut sum = 0.0
    for j = 0; j < n_samples; j = j + 1 {
      sum = sum + data[j][i]
    }
    self.mean[i] = sum / n_samples.to_double()
  }

  // 计算标准差
  for i = 0; i < n_features; i = i + 1 {
    let mut sum_sq = 0.0
    for j = 0; j < n_samples; j = j + 1 {
      let diff = data[j][i] - self.mean[i]
      sum_sq = sum_sq + diff * diff
    }
    self.std[i] = (sum_sq / n_samples.to_double()).sqrt()
    // 避免除以零
    if self.std[i] == 0.0 {
      self.std[i] = 1.0
    }
  }
  self.fitted = true
}

///|
/// 转换数据
/// @param data 待转换的数据
/// @return 标准化后的数据
pub fn StandardScaler::transform(
  self : StandardScaler,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let n_features = validate_feature_matrix(data)
  if n_features != self.mean.length() {
    abort("特征数量与拟合时不一致")
  }
  let n_samples = data.length()
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      row[j] = (data[i][j] - self.mean[j]) / self.std[j]
    }
    result[i] = row
  }
  result
}

///|
/// 拟合并转换数据（一步完成）
pub fn StandardScaler::fit_transform(
  self : StandardScaler,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  self.fit(data)
  self.transform(data)
}

///|
/// 反转换（还原标准化）
pub fn StandardScaler::inverse_transform(
  self : StandardScaler,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let n_features = validate_feature_matrix(data)
  if n_features != self.mean.length() {
    abort("特征数量与拟合时不一致")
  }
  let n_samples = data.length()
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      row[j] = data[i][j] * self.std[j] + self.mean[j]
    }
    result[i] = row
  }
  result
}

///|
/// 最小-最大缩放器 (MinMaxScaler)
/// 将特征缩放到指定范围（默认 [0, 1]）
/// 公式: x_scaled = (x - min) / (max - min) * (feature_max - feature_min) + feature_min
pub struct MinMaxScaler {
  mut data_min : Array[Double]
  mut data_max : Array[Double]
  feature_range_min : Double
  feature_range_max : Double
  mut fitted : Bool
}

///|
pub fn MinMaxScaler::new(
  feature_range? : (Double, Double) = (0.0, 1.0),
) -> MinMaxScaler {
  {
    data_min: [],
    data_max: [],
    feature_range_min: feature_range.0,
    feature_range_max: feature_range.1,
    fitted: false,
  }
}

///|
/// 拟合数据，计算最小值和最大值
pub fn MinMaxScaler::fit(
  self : MinMaxScaler,
  data : Array[Array[Double]],
) -> Unit {
  let n_features = validate_feature_matrix(data)
  let n_samples = data.length()
  self.data_min = Array::make(n_features, @double.infinity)
  self.data_max = Array::make(n_features, @double.neg_infinity)

  // 找出每个特征的最小值和最大值
  for i = 0; i < n_features; i = i + 1 {
    for j = 0; j < n_samples; j = j + 1 {
      let value = data[j][i]
      if value < self.data_min[i] {
        self.data_min[i] = value
      }
      if value > self.data_max[i] {
        self.data_max[i] = value
      }
    }
    // 避免除以零
    if self.data_min[i] == self.data_max[i] {
      self.data_max[i] = self.data_min[i] + 1.0
    }
  }
  self.fitted = true
}

///|
/// 转换数据
pub fn MinMaxScaler::transform(
  self : MinMaxScaler,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let n_features = validate_feature_matrix(data)
  if n_features != self.data_min.length() {
    abort("特征数量与拟合时不一致")
  }
  let n_samples = data.length()
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      let x_std = (data[i][j] - self.data_min[j]) /
        (self.data_max[j] - self.data_min[j])
      row[j] = x_std * (self.feature_range_max - self.feature_range_min) +
        self.feature_range_min
    }
    result[i] = row
  }
  result
}

///|
/// 拟合并转换数据
pub fn MinMaxScaler::fit_transform(
  self : MinMaxScaler,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  self.fit(data)
  self.transform(data)
}

///|
/// 反转换
pub fn MinMaxScaler::inverse_transform(
  self : MinMaxScaler,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let n_features = validate_feature_matrix(data)
  if n_features != self.data_min.length() {
    abort("特征数量与拟合时不一致")
  }
  let n_samples = data.length()
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      let x_std = (data[i][j] - self.feature_range_min) /
        (self.feature_range_max - self.feature_range_min)
      row[j] = x_std * (self.data_max[j] - self.data_min[j]) + self.data_min[j]
    }
    result[i] = row
  }
  result
}

///|
/// 归一化器 (Normalizer)
/// 将每个样本归一化为单位范数
/// 支持 L1, L2 和 Max 范数
pub(all) enum NormType {
  L1
  L2
  Max
} derive(Show, Eq)

///|
pub struct Normalizer {
  norm : NormType
}

///|
pub fn Normalizer::new(norm? : NormType = L2) -> Normalizer {
  { norm, }
}

///|
/// 转换数据（归一化不需要拟合）
pub fn Normalizer::transform(
  self : Normalizer,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  let n_samples = data.length()
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    result[i] = self.normalize_sample(data[i])
  }
  result
}

///|
/// 拟合并转换（归一化不需要拟合，直接转换）
pub fn Normalizer::fit_transform(
  self : Normalizer,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  self.transform(data)
}

///|
/// 归一化单个样本
fn Normalizer::normalize_sample(
  self : Normalizer,
  sample : Array[Double],
) -> Array[Double] {
  let n_features = sample.length()
  let result = Array::make(n_features, 0.0)

  // 计算范数
  let norm_value = match self.norm {
    L1 => {
      let mut sum = 0.0
      for i = 0; i < n_features; i = i + 1 {
        sum = sum + sample[i].abs()
      }
      sum
    }
    L2 => {
      let mut sum = 0.0
      for i = 0; i < n_features; i = i + 1 {
        sum = sum + sample[i] * sample[i]
      }
      sum.sqrt()
    }
    Max => {
      let mut max_val = 0.0
      for i = 0; i < n_features; i = i + 1 {
        let abs_val = sample[i].abs()
        if abs_val > max_val {
          max_val = abs_val
        }
      }
      max_val
    }
  }

  // 避免除以零
  let norm_value = if norm_value == 0.0 { 1.0 } else { norm_value }

  // 归一化
  for i = 0; i < n_features; i = i + 1 {
    result[i] = sample[i] / norm_value
  }
  result
}

///|
/// 鲁棒缩放器 (RobustScaler)
/// 使用中位数和四分位距进行缩放，对异常值更鲁棒
/// 公式: x_scaled = (x - median) / IQR
pub struct RobustScaler {
  mut median : Array[Double]
  mut iqr : Array[Double] // 四分位距 (Q3 - Q1)
  mut fitted : Bool
}

///|
pub fn RobustScaler::new() -> RobustScaler {
  { median: [], iqr: [], fitted: false }
}

///|
/// 拟合数据
pub fn RobustScaler::fit(
  self : RobustScaler,
  data : Array[Array[Double]],
) -> Unit {
  if data.length() == 0 {
    abort("数据不能为空")
  }
  let n_samples = data.length()
  let n_features = data[0].length()
  self.median = Array::make(n_features, 0.0)
  self.iqr = Array::make(n_features, 0.0)

  // 对每个特征计算中位数和IQR
  for i = 0; i < n_features; i = i + 1 {
    // 提取特征列并排序
    let feature_values = Array::make(n_samples, 0.0)
    for j = 0; j < n_samples; j = j + 1 {
      feature_values[j] = data[j][i]
    }
    feature_values.sort()

    // 计算中位数
    self.median[i] = calculate_median(feature_values)

    // 计算Q1和Q3
    let q1 = calculate_percentile(feature_values, 25.0)
    let q3 = calculate_percentile(feature_values, 75.0)
    self.iqr[i] = q3 - q1

    // 避免除以零
    if self.iqr[i] == 0.0 {
      self.iqr[i] = 1.0
    }
  }
  self.fitted = true
}

///|
/// 转换数据
pub fn RobustScaler::transform(
  self : RobustScaler,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let n_samples = data.length()
  let n_features = data[0].length()
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      row[j] = (data[i][j] - self.median[j]) / self.iqr[j]
    }
    result[i] = row
  }
  result
}

///|
/// 拟合并转换数据
pub fn RobustScaler::fit_transform(
  self : RobustScaler,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  self.fit(data)
  self.transform(data)
}

///|
/// 辅助函数：计算中位数
fn calculate_median(sorted_data : Array[Double]) -> Double {
  let n = sorted_data.length()
  if n % 2 == 1 {
    sorted_data[n / 2]
  } else {
    (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
  }
}

///|
/// 辅助函数：计算百分位数
fn calculate_percentile(
  sorted_data : Array[Double],
  percentile : Double,
) -> Double {
  let n = sorted_data.length()
  let index = percentile / 100.0 * (n - 1).to_double()
  let lower = index.to_int()
  let upper = lower + 1
  if upper >= n {
    sorted_data[n - 1]
  } else {
    let weight = index - lower.to_double()
    sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
  }
}
