///|
/// 内部统计小工具

///|
fn mean_array(values : Array[Double]) -> Double {
  if values.length() == 0 {
    abort("空数组无法计算均值")
  }
  let mut total = 0.0
  for i = 0; i < values.length(); i = i + 1 {
    total = total + values[i]
  }
  total / values.length().to_double()
}

///|
fn variance_array(values : Array[Double]) -> Double {
  let mu = mean_array(values)
  let mut sum_sq = 0.0
  for i = 0; i < values.length(); i = i + 1 {
    let diff = values[i] - mu
    sum_sq = sum_sq + diff * diff
  }
  sum_sq / values.length().to_double()
}
