///|
/// StandardScaler 测试
test "StandardScaler - 基本功能" {
  let scaler : StandardScaler = StandardScaler::new()
  let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  let transformed = scaler.fit_transform(data)

  // 验证形状
  assert_eq(transformed.length(), 3)
  assert_eq(transformed[0].length(), 3)

  // 标准化后的均值应接近0，标准差应接近1
  // 第一列: [1, 4, 7], 均值=4, 标准差约为2.45
  assert_true((transformed[0][0] + 1.224744871391589).abs() < 0.0001)
  assert_true((transformed[1][0] - 0.0).abs() < 0.0001)
  assert_true((transformed[2][0] - 1.224744871391589).abs() < 0.0001)
}

///|
test "StandardScaler - 反转换" {
  let scaler = StandardScaler::new()
  let data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
  let transformed = scaler.fit_transform(data)
  let inverse = scaler.inverse_transform(transformed)

  // 验证反转换后恢复原始值
  for i = 0; i < data.length(); i = i + 1 {
    for j = 0; j < data[i].length(); j = j + 1 {
      assert_true((data[i][j] - inverse[i][j]).abs() < 0.0001)
    }
  }
}

///|
/// MinMaxScaler 测试
test "MinMaxScaler - 默认范围 [0, 1]" {
  let scaler = MinMaxScaler::new()
  let data = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
  let transformed = scaler.fit_transform(data)

  // 验证缩放到 [0, 1]
  assert_true((transformed[0][0] - 0.0).abs() < 0.0001)
  assert_true((transformed[2][0] - 1.0).abs() < 0.0001)
  assert_true((transformed[0][1] - 0.0).abs() < 0.0001)
  assert_true((transformed[2][1] - 1.0).abs() < 0.0001)
}

///|
test "MinMaxScaler - 自定义范围 [-1, 1]" {
  let scaler = MinMaxScaler::new(feature_range=(-1.0, 1.0))
  let data = [[0.0], [5.0], [10.0]]
  let transformed = scaler.fit_transform(data)
  assert_true((transformed[0][0] - -1.0).abs() < 0.0001)
  assert_true((transformed[1][0] - 0.0).abs() < 0.0001)
  assert_true((transformed[2][0] - 1.0).abs() < 0.0001)
}

///|
test "MinMaxScaler - 反转换" {
  let scaler = MinMaxScaler::new()
  let data = [[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]]
  let transformed = scaler.fit_transform(data)
  let inverse = scaler.inverse_transform(transformed)
  for i = 0; i < data.length(); i = i + 1 {
    for j = 0; j < data[i].length(); j = j + 1 {
      assert_true((data[i][j] - inverse[i][j]).abs() < 0.0001)
    }
  }
}

///|
/// Normalizer 测试
test "Normalizer - L2 范数" {
  let normalizer = Normalizer::new(norm=L2)
  let data = [
    [3.0, 4.0], // 范数为 5
    [5.0, 12.0],
  ] // 范数为 13
  let transformed = normalizer.transform(data)

  // 验证 L2 范数为 1
  let norm1 = (transformed[0][0] * transformed[0][0] +
  transformed[0][1] * transformed[0][1]).sqrt()
  let norm2 = (transformed[1][0] * transformed[1][0] +
  transformed[1][1] * transformed[1][1]).sqrt()
  assert_true((norm1 - 1.0).abs() < 0.0001)
  assert_true((norm2 - 1.0).abs() < 0.0001)

  // 验证具体值
  assert_true((transformed[0][0] - 0.6).abs() < 0.0001)
  assert_true((transformed[0][1] - 0.8).abs() < 0.0001)
}

///|
test "Normalizer - L1 范数" {
  let normalizer = Normalizer::new(norm=L1)
  let data = [[3.0, 6.0]] // L1 范数为 9
  let transformed = normalizer.transform(data)

  // 验证 L1 范数为 1
  let norm = transformed[0][0].abs() + transformed[0][1].abs()
  assert_true((norm - 1.0).abs() < 0.0001)
  assert_true((transformed[0][0] - 1.0 / 3.0).abs() < 0.0001)
  assert_true((transformed[0][1] - 2.0 / 3.0).abs() < 0.0001)
}

///|
test "Normalizer - Max 范数" {
  let normalizer = Normalizer::new(norm=Max)
  let data = [[2.0, 8.0, 4.0]] // 最大值为 8
  let transformed = normalizer.transform(data)
  assert_true((transformed[0][0] - 0.25).abs() < 0.0001)
  assert_true((transformed[0][1] - 1.0).abs() < 0.0001)
  assert_true((transformed[0][2] - 0.5).abs() < 0.0001)
}

///|
test "Normalizer - fit_transform also normalizes" {
  let normalizer = Normalizer::new(norm=L2)
  let data = [[0.0, 0.0], [1.0, 0.0]]
  let transformed = normalizer.fit_transform(data)
  // 第一行全零，第二行范数应为 1
  assert_eq(transformed.length(), 2)
  let norm = (transformed[1][0] * transformed[1][0] +
  transformed[1][1] * transformed[1][1]).sqrt()
  assert_true((norm - 1.0).abs() < 1.0e-9)
}

///|
/// RobustScaler 测试
test "RobustScaler - 基本功能" {
  let scaler = RobustScaler::new()
  let data = [[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]] // 异常值
  scaler.fit(data)
  let transformed = scaler.transform(data)

  // 验证中位数和IQR的计算
  // 中位数应该在3和4之间
  // RobustScaler 对异常值应该更鲁棒
  assert_eq(transformed.length(), 6)

  // 异常值100应该被缩放，但不会像StandardScaler那样极端
  assert_true(transformed[5][0] > 0.0)
}

///|
test "RobustScaler - 拟合转换" {
  let scaler = RobustScaler::new()
  let data = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]
  let transformed = scaler.fit_transform(data)
  assert_eq(transformed.length(), 5)
  assert_eq(transformed[0].length(), 2)
}

///|
test "RobustScaler - handles zero IQR by fallback" {
  let scaler = RobustScaler::new()
  let data = [[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]]
  let transformed = scaler.fit_transform(data)
  // 所有值相同，IQR 为 0，会 fallback 为 1，输出应为 0
  for i = 0; i < transformed.length(); i = i + 1 {
    for j = 0; j < transformed[i].length(); j = j + 1 {
      assert_eq(transformed[i][j], 0.0)
    }
  }
}
