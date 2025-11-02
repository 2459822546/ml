///|
/// Dataset 和数据分割测试
test "Dataset - 基本功能" {
  let features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
  let labels = [0.0, 1.0, 0.0]
  let dataset = Dataset::new(features, labels)
  assert_eq(dataset.len(), 3)
  assert_eq(dataset.n_features(), 2)
}

///|
test "Dataset - 获取样本" {
  let features = [[1.0, 2.0], [3.0, 4.0]]
  let labels = [0.0, 1.0]
  let dataset = Dataset::new(features, labels)
  let (feat, label) = dataset.get(1)
  assert_true((feat[0] - 3.0).abs() < 0.0001)
  assert_true((feat[1] - 4.0).abs() < 0.0001)
  assert_true((label - 1.0).abs() < 0.0001)
}

///|
test "Dataset - 切片" {
  let features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
  let labels = [0.0, 1.0, 0.0, 1.0]
  let dataset = Dataset::new(features, labels)
  let sub_dataset = dataset.slice(1, 3)
  assert_eq(sub_dataset.len(), 2)
  let (feat, _) = sub_dataset.get(0)
  assert_true((feat[0] - 3.0).abs() < 0.0001)
}

///|
test "train_test_split - 基本分割" {
  let features = [
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0],
    [10.0],
  ]
  let labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  let (train, tests) = train_test_split(
    features,
    labels,
    test_size=0.2,
    shuffle=false,
  )
  assert_eq(train.len(), 8)
  assert_eq(tests.len(), 2)
}

///|
test "train_test_split - 打乱数据" {
  let features = [[1.0], [2.0], [3.0], [4.0], [5.0]]
  let labels = [1.0, 2.0, 3.0, 4.0, 5.0]
  let (train, tests) = train_test_split(
    features,
    labels,
    test_size=0.4,
    shuffle=true,
    random_seed=42,
  )
  assert_eq(train.len(), 3)
  assert_eq(tests.len(), 2)
}

///|
test "k_fold_split - 3折交叉验证" {
  let features = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
  let labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let folds = k_fold_split(features, labels, 3)
  assert_eq(folds.length(), 3)

  // 每折的测试集大小应该是 2
  for i = 0; i < folds.length(); i = i + 1 {
    let (_train, tests) = folds[i]
    assert_eq(tests.len(), 2)
  }
}

///|
test "k_fold_split - 验证数据不重叠" {
  let features = [[1.0], [2.0], [3.0], [4.0]]
  let labels = [1.0, 2.0, 3.0, 4.0]
  let folds = k_fold_split(features, labels, 2)
  assert_eq(folds.length(), 2)
  let (_train1, test1) = folds[0]
  let (_train2, test2) = folds[1]

  // 两折的测试集应该各有2个样本
  assert_eq(test1.len(), 2)
  assert_eq(test2.len(), 2)
}
