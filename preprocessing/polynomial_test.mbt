///|
/// PolynomialFeatures 测试
test "PolynomialFeatures - 二次多项式，包含偏置" {
  let poly = PolynomialFeatures::new(degree=2, include_bias=true)
  let data = [[2.0, 3.0]]
  let transformed = poly.fit_transform(data)

  // [1, a, b, a^2, ab, b^2]
  // [1, 2, 3, 4, 6, 9]
  assert_eq(transformed.length(), 1)
  assert_eq(transformed[0].length(), 6)
  assert_true((transformed[0][0] - 1.0).abs() < 0.0001) // 偏置
  assert_true((transformed[0][1] - 2.0).abs() < 0.0001) // a
  assert_true((transformed[0][2] - 3.0).abs() < 0.0001) // b
  assert_true((transformed[0][3] - 4.0).abs() < 0.0001) // a^2
  assert_true((transformed[0][4] - 6.0).abs() < 0.0001) // ab
  assert_true((transformed[0][5] - 9.0).abs() < 0.0001) // b^2
}

///|
test "PolynomialFeatures - 二次多项式，不包含偏置" {
  let poly = PolynomialFeatures::new(degree=2, include_bias=false)
  let data = [[2.0, 3.0]]
  let transformed = poly.fit_transform(data)

  // [a, b, a^2, ab, b^2]
  assert_eq(transformed[0].length(), 5)
  assert_true((transformed[0][0] - 2.0).abs() < 0.0001)
  assert_true((transformed[0][1] - 3.0).abs() < 0.0001)
  assert_true((transformed[0][2] - 4.0).abs() < 0.0001)
  assert_true((transformed[0][3] - 6.0).abs() < 0.0001)
  assert_true((transformed[0][4] - 9.0).abs() < 0.0001)
}

///|
test "PolynomialFeatures - 三次多项式" {
  let poly = PolynomialFeatures::new(degree=3, include_bias=false)
  let data = [[2.0, 3.0]]
  let transformed = poly.fit_transform(data)

  // 应该包含: a, b, a^2, ab, b^2, a^3, a^2*b, a*b^2, b^3
  // 总共 9 个特征
  assert_eq(transformed[0].length(), 9)
}

///|
test "PolynomialFeatures - 仅交互特征" {
  let poly = PolynomialFeatures::new(
    degree=2,
    include_bias=false,
    interaction_only=true,
  )
  let data = [[2.0, 3.0]]
  let transformed = poly.fit_transform(data)

  // 仅交互: [a, b, ab]（不包含 a^2 和 b^2）
  assert_eq(transformed[0].length(), 3)
  assert_true((transformed[0][0] - 2.0).abs() < 0.0001) // a
  assert_true((transformed[0][1] - 3.0).abs() < 0.0001) // b
  assert_true((transformed[0][2] - 6.0).abs() < 0.0001) // ab
}

///|
test "PolynomialFeatures - 单特征二次" {
  let poly = PolynomialFeatures::new(degree=2, include_bias=true)
  let data = [[5.0]]
  let transformed = poly.fit_transform(data)

  // [1, x, x^2]
  assert_eq(transformed[0].length(), 3)
  assert_true((transformed[0][0] - 1.0).abs() < 0.0001)
  assert_true((transformed[0][1] - 5.0).abs() < 0.0001)
  assert_true((transformed[0][2] - 25.0).abs() < 0.0001)
}

///|
test "PolynomialFeatures - 多样本" {
  let poly = PolynomialFeatures::new(degree=2, include_bias=false)
  let data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
  let transformed = poly.fit_transform(data)
  assert_eq(transformed.length(), 3)

  // 验证第一个样本: [1, 2, 1, 2, 4]
  assert_true((transformed[0][0] - 1.0).abs() < 0.0001)
  assert_true((transformed[0][1] - 2.0).abs() < 0.0001)
  assert_true((transformed[0][2] - 1.0).abs() < 0.0001)
  assert_true((transformed[0][3] - 2.0).abs() < 0.0001)
  assert_true((transformed[0][4] - 4.0).abs() < 0.0001)

  // 验证第二个样本: [3, 4, 9, 12, 16]
  assert_true((transformed[1][0] - 3.0).abs() < 0.0001)
  assert_true((transformed[1][1] - 4.0).abs() < 0.0001)
  assert_true((transformed[1][2] - 9.0).abs() < 0.0001)
  assert_true((transformed[1][3] - 12.0).abs() < 0.0001)
  assert_true((transformed[1][4] - 16.0).abs() < 0.0001)
}

///|
test "PolynomialFeatures - 获取输出特征数" {
  let poly = PolynomialFeatures::new(degree=2, include_bias=true)
  let data = [[1.0, 2.0, 3.0]]
  poly.fit(data)
  let n_features = poly.get_n_output_features()

  // 3个输入特征，度数2，包含偏置
  // 应该是: 1 + 3 + 6 = 10 个特征
  // (偏置) + (a,b,c) + (a^2,ab,ac,b^2,bc,c^2)
  assert_eq(n_features, 10)
}
