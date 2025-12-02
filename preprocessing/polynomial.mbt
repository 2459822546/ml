///|
/// 多项式特征生成器 (PolynomialFeatures)
/// 生成多项式特征和交互特征
/// 例如: [a, b] -> [1, a, b, a^2, ab, b^2] (degree=2)
pub struct PolynomialFeatures {
  degree : Int
  include_bias : Bool
  interaction_only : Bool
  mut n_input_features : Int
  mut n_output_features : Int
  mut fitted : Bool
}

///|
pub fn PolynomialFeatures::new(
  degree? : Int = 2,
  include_bias? : Bool = true,
  interaction_only? : Bool = false,
) -> PolynomialFeatures {
  if degree < 1 {
    abort("degree 必须至少为 1")
  }
  {
    degree,
    include_bias,
    interaction_only,
    n_input_features: 0,
    n_output_features: 0,
    fitted: false,
  }
}

///|
/// 拟合数据（主要是记录特征数量）
pub fn PolynomialFeatures::fit(
  self : PolynomialFeatures,
  data : Array[Array[Double]],
) -> Unit {
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
  self.n_input_features = n_features
  self.n_output_features = self.calculate_n_output_features()
  self.fitted = true
}

///|
/// 转换数据生成多项式特征
pub fn PolynomialFeatures::transform(
  self : PolynomialFeatures,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let n_samples = data.length()
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    result[i] = self.transform_sample(data[i])
  }
  result
}

///|
/// 拟合并转换
pub fn PolynomialFeatures::fit_transform(
  self : PolynomialFeatures,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  self.fit(data)
  self.transform(data)
}

///|
/// 转换单个样本
fn PolynomialFeatures::transform_sample(
  self : PolynomialFeatures,
  sample : Array[Double],
) -> Array[Double] {
  let features = Array::new()

  // 添加偏置项
  if self.include_bias {
    features.push(1.0)
  }

  // 生成特征组合
  if self.interaction_only {
    self.generate_interaction_features(sample, features)
  } else {
    self.generate_polynomial_features(sample, features)
  }

  // 转换为固定数组
  let result = Array::make(features.length(), 0.0)
  for i = 0; i < features.length(); i = i + 1 {
    result[i] = features[i]
  }
  result
}

///|
/// 生成多项式特征（包括纯次幂）
fn PolynomialFeatures::generate_polynomial_features(
  self : PolynomialFeatures,
  sample : Array[Double],
  features : Array[Double],
) -> Unit {
  // 生成所有次数的组合
  for d = 1; d <= self.degree; d = d + 1 {
    self.generate_combinations(sample, features, d, 0, 1.0, 0)
  }
}

///|
/// 递归生成特征组合
fn PolynomialFeatures::generate_combinations(
  self : PolynomialFeatures,
  sample : Array[Double],
  features : Array[Double],
  degree : Int,
  start_index : Int,
  current_value : Double,
  current_degree : Int,
) -> Unit {
  if current_degree == degree {
    features.push(current_value)
    return
  }
  for i = start_index; i < sample.length(); i = i + 1 {
    self.generate_combinations(
      sample,
      features,
      degree,
      i,
      current_value * sample[i],
      current_degree + 1,
    )
  }
}

///|
/// 生成交互特征（不包括纯次幂）
fn PolynomialFeatures::generate_interaction_features(
  self : PolynomialFeatures,
  sample : Array[Double],
  features : Array[Double],
) -> Unit {
  let n = sample.length()

  // 添加一次特征
  for i = 0; i < n; i = i + 1 {
    features.push(sample[i])
  }

  // 添加交互特征
  for d = 2; d <= self.degree; d = d + 1 {
    self.generate_interaction_combinations(sample, features, d, 0, 1.0, 0)
  }
}

///|
/// 递归生成交互特征组合（确保不同特征的组合）
fn PolynomialFeatures::generate_interaction_combinations(
  self : PolynomialFeatures,
  sample : Array[Double],
  features : Array[Double],
  degree : Int,
  start_index : Int,
  current_value : Double,
  current_degree : Int,
) -> Unit {
  if current_degree == degree {
    features.push(current_value)
    return
  }
  for i = start_index; i < sample.length(); i = i + 1 {
    self.generate_interaction_combinations(
      sample,
      features,
      degree,
      i + 1, // 确保不重复使用同一特征
      current_value * sample[i],
      current_degree + 1,
    )
  }
}

///|
/// 计算输出特征数量
fn PolynomialFeatures::calculate_n_output_features(
  self : PolynomialFeatures,
) -> Int {
  let n = self.n_input_features
  let d = self.degree
  let mut total = if self.include_bias { 1 } else { 0 }
  if self.interaction_only {
    // 只包含交互特征
    for deg = 1; deg <= d; deg = deg + 1 {
      total = total + binomial_coefficient(n, deg)
    }
  } else {
    // 包含所有多项式特征
    total = total + binomial_coefficient(n + d, d) - 1
    if not(self.include_bias) {
      total = total + 1
    }
  }
  total
}

///|
/// 计算二项式系数 C(n, k) = n! / (k! * (n-k)!)
fn binomial_coefficient(n : Int, k : Int) -> Int {
  if k > n {
    return 0
  }
  if k == 0 || k == n {
    return 1
  }
  let k = if k > n - k { n - k } else { k }
  let mut result = 1
  for i = 0; i < k; i = i + 1 {
    result = result * (n - i) / (i + 1)
  }
  result
}

///|
/// 获取输出特征数量
pub fn PolynomialFeatures::get_n_output_features(
  self : PolynomialFeatures,
) -> Int {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  self.n_output_features
}
