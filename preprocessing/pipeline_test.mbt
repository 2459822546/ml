///|
/// 流水线与列变换器测试

///|
test "Pipeline with StandardScaler then PolynomialFeatures" {
  let pipeline = Pipeline::new()
  pipeline.add_step("scale", transformer_standard(StandardScaler::new()))
  pipeline.add_step(
    "poly",
    transformer_polynomial(
      PolynomialFeatures::new(degree=2, include_bias=false),
    ),
  )
  let x = [[1.0, 2.0], [3.0, 4.0]]
  let out = pipeline.fit_transform(x)
  // 二次多项式（无偏置）: [x1, x2, x1^2, x1*x2, x2^2]
  assert_eq(out.length(), 2)
  assert_eq(out[0].length(), 5)
  // 缩放后的前两列均值应接近 0
  let mean0 = (out[0][0] + out[1][0]) / 2.0
  let mean1 = (out[0][1] + out[1][1]) / 2.0
  assert_true(mean0.abs() < 1.0e-6)
  assert_true(mean1.abs() < 1.0e-6)
}

///|
test "ColumnTransformer combine scaled block and passthrough" {
  let ct = ColumnTransformer::new(remainder_passthrough=true)
  ct.add("scale_first_two", [0, 1], transformer_standard(StandardScaler::new()))
  let data = [[1.0, 10.0, 100.0], [3.0, 30.0, 300.0], [5.0, 50.0, 500.0]]
  let out = ct.fit_transform(data)
  // 输出包含 2 个缩放列 + 1 个透传列
  assert_eq(out.length(), 3)
  assert_eq(out[0].length(), 3)
  // 透传列应保留原值
  assert_eq(out[0][2], 100.0)
  assert_eq(out[1][2], 300.0)
  assert_eq(out[2][2], 500.0)
}

///|
test "ColumnTransformer drop remainder and scale subset" {
  let ct = ColumnTransformer::new(remainder_passthrough=false)
  ct.add("minmax_last", [2], transformer_minmax(MinMaxScaler::new()))
  let data = [[1.0, 2.0, 10.0], [1.0, 3.0, 20.0], [1.0, 4.0, 30.0]]
  let out = ct.fit_transform(data)
  // 只保留一个经过 MinMax 的列
  assert_eq(out.length(), 3)
  assert_eq(out[0].length(), 1)
  assert_true((out[0][0] - 0.0).abs() < 1.0e-9)
  assert_true((out[2][0] - 1.0).abs() < 1.0e-9)
}
