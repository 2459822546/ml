///|
/// Stats 测试
test "Stats - 中位数" {
  let data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
  assert_eq(median(data1), 3.0)
  let data2 = [1.0, 2.0, 3.0, 4.0]
  assert_eq(median(data2), 2.5)
}

///|
test "Stats - 百分位数" {
  let data = [1.0, 2.0, 3.0, 4.0, 5.0]
  assert_eq(percentile(data, 0.0), 1.0)
  assert_eq(percentile(data, 100.0), 5.0)
  assert_eq(percentile(data, 50.0), 3.0)
}

///|
test "Stats - IQR" {
  let data = [1.0, 2.0, 3.0, 4.0, 5.0]
  let iqr_val = iqr(data)

  // Q1 = 2, Q3 = 4, IQR = 2
  assert_eq(iqr_val, 2.0)
}

///|
test "Stats - 协方差" {
  let x = [1.0, 2.0, 3.0, 4.0, 5.0]
  let y = [2.0, 4.0, 6.0, 8.0, 10.0]
  let cov = covariance(x, y)
  // y = 2x, 所以协方差应该是 2 * variance(x)
  assert_true(cov > 0.0)
}

///|
test "Stats - 相关系数" {
  let x = [1.0, 2.0, 3.0, 4.0, 5.0]
  let y = [2.0, 4.0, 6.0, 8.0, 10.0]
  let corr = correlation(x, y)
  // 完全线性关系，相关系数应该是 1
  assert_true((corr - 1.0).abs() < 0.0001)

  // 负相关
  let y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
  let corr_neg = correlation(x, y_neg)
  assert_true((corr_neg + 1.0).abs() < 0.0001)
}

///|
test "Stats - 欧几里得距离" {
  let x = [1.0, 2.0, 3.0]
  let y = [4.0, 6.0, 8.0]
  let dist = euclidean_distance(x, y)
  // sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
  let expected = 50.0.sqrt()
  assert_true((dist - expected).abs() < 0.0001)
}

///|
test "Stats - 曼哈顿距离" {
  let x = [1.0, 2.0, 3.0]
  let y = [4.0, 6.0, 8.0]
  let dist = manhattan_distance(x, y)
  // |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
  assert_eq(dist, 12.0)
}

///|
test "Stats - 余弦相似度" {
  let x = [1.0, 2.0, 3.0]
  let y = [2.0, 4.0, 6.0]
  let sim = cosine_similarity(x, y)
  // 完全同方向，相似度应该是 1
  assert_true((sim - 1.0).abs() < 0.0001)

  // 垂直向量
  let x2 = [1.0, 0.0]
  let y2 = [0.0, 1.0]
  let sim2 = cosine_similarity(x2, y2)
  assert_true(sim2.abs() < 0.0001)
}


