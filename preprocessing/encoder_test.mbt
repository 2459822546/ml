///|
/// LabelEncoder 测试
test "LabelEncoder - 基本功能" {
  let encoder = LabelEncoder::new()
  let labels = ["cat", "dog", "cat", "bird", "dog", "bird"]
  let encoded = encoder.fit_transform(labels)

  // 验证编码
  assert_eq(encoded.length(), 6)

  // 相同的标签应该有相同的编码
  assert_eq(encoded[0], encoded[2]) // cat
  assert_eq(encoded[1], encoded[4]) // dog
  assert_eq(encoded[3], encoded[5]) // bird

  // 不同的标签应该有不同的编码
  assert_true(encoded[0] != encoded[1])
  assert_true(encoded[0] != encoded[3])
  assert_true(encoded[1] != encoded[3])
}

///|
test "LabelEncoder - 反转换" {
  let encoder = LabelEncoder::new()
  let labels = ["apple", "banana", "cherry"]
  let encoded = encoder.fit_transform(labels)
  let decoded = encoder.inverse_transform(encoded)

  // 验证反转换后恢复原始标签
  for i = 0; i < labels.length(); i = i + 1 {
    assert_eq(labels[i], decoded[i])
  }
}

///|
test "LabelEncoder - 获取类别" {
  let encoder = LabelEncoder::new()
  let labels = ["red", "green", "blue", "red", "green"]
  encoder.fit(labels)
  let classes = encoder.get_classes()
  println(classes)

  // 类别应该是排序后的唯一值
  assert_eq(classes.length(), 3)
  assert_eq(classes[0], "blue")
  assert_eq(classes[1], "green")
  assert_eq(classes[2], "red")
}

///|
/// OneHotEncoder 测试
test "OneHotEncoder - 单特征" {
  let encoder = OneHotEncoder::new()
  let data = [["red"], ["green"], ["blue"], ["red"]]
  let encoded = encoder.fit_transform(data)

  // 3个类别，所以应该有3列
  assert_eq(encoded.length(), 4)
  assert_eq(encoded[0].length(), 3)

  // 验证独热编码：每行只有一个1
  for i = 0; i < encoded.length(); i = i + 1 {
    let mut sum = 0.0
    for j = 0; j < encoded[i].length(); j = j + 1 {
      sum = sum + encoded[i][j]
    }
    assert_true((sum - 1.0).abs() < 0.0001)
  }

  // 相同的值应该有相同的编码
  for j = 0; j < 3; j = j + 1 {
    assert_eq(encoded[0][j], encoded[3][j])
  }
}

///|
test "OneHotEncoder - 多特征" {
  let encoder = OneHotEncoder::new()
  let data = [["male", "US"], ["female", "UK"], ["male", "UK"]]
  let encoded = encoder.fit_transform(data)

  // 性别2类 + 国家2类 = 4列
  assert_eq(encoded.length(), 3)
  assert_eq(encoded[0].length(), 4)
}

///|
test "OneHotEncoder - 特征名称" {
  let encoder = OneHotEncoder::new()
  let data = [["A", "X"], ["B", "Y"]]
  encoder.fit(data)
  let feature_names = encoder.get_feature_names()

  // 应该有 4 个特征名称
  assert_eq(feature_names.length(), 4)
}

///|
/// Binarizer 测试
test "Binarizer - 默认阈值 0" {
  let binarizer = Binarizer::new()
  let data = [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]
  let binary = binarizer.transform(data)
  assert_true((binary[0][0] - 0.0).abs() < 0.0001)
  assert_true((binary[0][1] - 0.0).abs() < 0.0001)
  assert_true((binary[0][2] - 1.0).abs() < 0.0001)
  assert_true((binary[1][0] - 1.0).abs() < 0.0001)
  assert_true((binary[1][1] - 0.0).abs() < 0.0001)
  assert_true((binary[1][2] - 1.0).abs() < 0.0001)
}

///|
test "Binarizer - 自定义阈值" {
  let binarizer = Binarizer::new(threshold=5.0)
  let data = [[3.0, 7.0, 5.0], [10.0, 2.0, 5.1]]
  let binary = binarizer.transform(data)
  assert_true((binary[0][0] - 0.0).abs() < 0.0001)
  assert_true((binary[0][1] - 1.0).abs() < 0.0001)
  assert_true((binary[0][2] - 0.0).abs() < 0.0001)
  assert_true((binary[1][0] - 1.0).abs() < 0.0001)
  assert_true((binary[1][1] - 0.0).abs() < 0.0001)
  assert_true((binary[1][2] - 1.0).abs() < 0.0001)
}
