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
test "Dataset - from_samples" {
  let samples = [Sample::new([1.0, 2.0], 0.0), Sample::new([3.0, 4.0], 1.0)]
  let dataset = Dataset::from_samples(samples)
  assert_eq(dataset.len(), 2)
  let (feat, label) = dataset.get(0)
  assert_true((feat[0] - 1.0).abs() < 0.0001)
  assert_true((label - 0.0).abs() < 0.0001)
  let roundtrip = dataset.to_samples()
  assert_eq(roundtrip.length(), 2)
  assert_eq(roundtrip[1].features[0], 3.0)
  assert_eq(roundtrip[1].label, 1.0)
}

///|
test "Dataset - get_batch" {
  let features = [[1.0], [2.0], [3.0]]
  let labels = [1.0, 2.0, 3.0]
  let dataset = Dataset::new(features, labels)
  let (batch_features, batch_labels) = dataset.get_batch([2, 0])
  assert_eq(batch_features.length(), 2)
  assert_eq(batch_features[0][0], 3.0)
  assert_eq(batch_features[1][0], 1.0)
  assert_eq(batch_labels[0], 3.0)
  assert_eq(batch_labels[1], 1.0)
}

///|
test "MultiDataset - 基本功能" {
  let features = [[1.0, 2.0], [3.0, 4.0]]
  let labels = [[0.0, 1.0], [1.0, 0.0]]
  let dataset = MultiDataset::new(features, labels)
  assert_eq(dataset.len(), 2)
  assert_eq(dataset.n_features(), 2)
  assert_eq(dataset.n_labels(), 2)
  let (feat, lab) = dataset.get(1)
  assert_true((feat[0] - 3.0).abs() < 0.0001)
  assert_true((lab[0] - 1.0).abs() < 0.0001)
}

///|
test "MultiDataset - from_samples and get_batch" {
  let samples = [
    MultiSample::new([1.0], [0.0, 1.0]),
    MultiSample::new([2.0], [1.0, 0.0]),
    MultiSample::new([3.0], [1.0, 1.0]),
  ]
  let dataset = MultiDataset::from_samples(samples)
  let (batch_features, batch_labels) = dataset.get_batch([2, 0])
  assert_eq(batch_features.length(), 2)
  assert_eq(batch_features[0][0], 3.0)
  assert_eq(batch_labels[1][1], 1.0)
  let roundtrip = dataset.to_samples()
  assert_eq(roundtrip.length(), 3)
  assert_eq(roundtrip[2].labels[1], 1.0)
}

///|
test "MultiDataset - shuffle" {
  let features = [[1.0], [2.0], [3.0], [4.0]]
  let labels = [[1.0], [2.0], [3.0], [4.0]]
  let dataset = MultiDataset::new(features, labels)
  let shuffled = dataset.shuffle(42)
  let indices = shuffled_indices(4, 42)
  for i = 0; i < 4; i = i + 1 {
    let (feat, lab) = shuffled.get(i)
    assert_eq(feat[0], features[indices[i]][0])
    assert_eq(lab[0], labels[indices[i]][0])
  }
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
test "Dataset - range" {
  let features = [[1.0], [2.0], [3.0]]
  let labels = [1.0, 2.0, 3.0]
  let dataset = Dataset::new(features, labels)
  let ranged = dataset.range(0, 2)
  assert_eq(ranged.len(), 2)
  let (feat, label) = ranged.get(1)
  assert_true((feat[0] - 2.0).abs() < 0.0001)
  assert_true((label - 2.0).abs() < 0.0001)
}

///|
test "Dataset - map" {
  let features = [[1.0, 2.0], [3.0, 4.0]]
  let labels = [1.0, 2.0]
  let dataset = Dataset::new(features, labels)
  let mapped = dataset.map(fn(feat, label) {
    ([feat[0] * 2.0, feat[1] * 2.0], label + 1.0)
  })
  let (feat0, label0) = mapped.get(0)
  assert_true((feat0[0] - 2.0).abs() < 0.0001)
  assert_true((feat0[1] - 4.0).abs() < 0.0001)
  assert_true((label0 - 2.0).abs() < 0.0001)
}

///|
test "Dataset - shuffle" {
  let features = [[1.0], [2.0], [3.0], [4.0]]
  let labels = [1.0, 2.0, 3.0, 4.0]
  let dataset = Dataset::new(features, labels)
  let shuffled = dataset.shuffle(42)
  let indices = shuffled_indices(4, 42)
  for i = 0; i < 4; i = i + 1 {
    let (feat, label) = shuffled.get(i)
    assert_eq(feat[0], features[indices[i]][0])
    assert_eq(label, labels[indices[i]])
  }
}

///|
fn shuffled_indices(n : Int, seed : Int) -> Array[Int] {
  let indices = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    indices[i] = i
  }
  let mut rng_state = seed
  for i = n - 1; i > 0; i = i - 1 {
    rng_state = (rng_state * 1103515245 + 12345) % 2147483647
    let j = (rng_state % (i + 1)).abs()
    let temp = indices[i]
    indices[i] = indices[j]
    indices[j] = temp
  }
  indices
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
