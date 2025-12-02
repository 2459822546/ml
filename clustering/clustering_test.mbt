///|
/// 聚类算法测试

///|
test "KMeans - 基础聚类" {
  // 创建简单的2D数据：两个明显的簇
  let x = @math.Matrix::new([
    [1.0, 1.0],
    [1.5, 2.0],
    [3.0, 4.0],
    [5.0, 7.0],
    [3.5, 5.0],
    [4.5, 5.0],
    [3.5, 4.5],
  ])
  let model = KMeans::new(2, 100, 1.0e-4, InitMethod::Random)
  model.fit(x)
  assert_true(model.is_fitted)
  let labels = model.labels
  assert_eq(labels.length(), 7)

  // 检查惯性大于0
  assert_true(model.get_inertia() > 0.0)
}

///|
test "KMeans - K-Means++ 初始化" {
  let x = @math.Matrix::new([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [10.0, 10.0],
    [10.0, 11.0],
    [11.0, 10.0],
  ])
  let model = KMeans::new(2, 100, 1.0e-4, InitMethod::KMeansPlusPlus)
  let labels = model.fit_predict(x)
  assert_eq(labels.length(), 6)

  // 前3个样本应该在一个簇，后3个在另一个簇
  assert_eq(labels[0], labels[1])
  assert_eq(labels[0], labels[2])
  assert_eq(labels[3], labels[4])
  assert_eq(labels[3], labels[5])
}

///|
test "KMeans - predict" {
  let x_train = @math.Matrix::new([
    [1.0, 1.0],
    [1.5, 2.0],
    [5.0, 7.0],
    [4.5, 5.0],
  ])
  let model = KMeans::new(2, 100, 1.0e-4, InitMethod::Random)
  model.fit(x_train)
  let x_test = @math.Matrix::new([[1.2, 1.5], [5.1, 6.5]])
  let predictions = model.predict(x_test)
  assert_eq(predictions.length(), 2)
}

///|
test "DBSCAN - 基础聚类" {
  let x = @math.Matrix::new([
    [1.0, 1.0],
    [1.5, 2.0],
    [1.3, 1.5],
    [5.0, 7.0],
    [5.2, 7.3],
    [5.1, 6.8],
    [10.0, 10.0], // 噪声点
  ])
  let model = DBSCAN::new(1.0, 2)
  let labels = model.fit_predict(x)
  assert_eq(labels.length(), 7)

  // 应该找到2个簇
  let n_clusters = model.get_n_clusters()
  assert_true(n_clusters >= 2)

  // 最后一个点应该是噪声（标签 -1）
  assert_eq(labels[6], -1)
}

///|
test "DBSCAN - 所有点在一个簇" {
  let x = @math.Matrix::new([[1.0, 1.0], [1.2, 1.1], [1.1, 1.2], [1.3, 1.0]])
  let model = DBSCAN::new(0.5, 2)
  model.fit(x)
  let labels = model.get_labels()

  // 所有点应该在同一个簇
  assert_eq(labels[0], labels[1])
  assert_eq(labels[0], labels[2])
  assert_eq(labels[0], labels[3])
}

///|
test "AgglomerativeClustering - 单链接" {
  let x = @math.Matrix::new([[1.0, 1.0], [1.5, 2.0], [5.0, 7.0], [4.5, 5.0]])
  let model = AgglomerativeClustering::new(2, LinkageMethod::Single)
  let labels = model.fit_predict(x)
  assert_eq(labels.length(), 4)

  // 应该有2个簇
  let unique_labels = []
  for i = 0; i < labels.length(); i = i + 1 {
    let label = labels[i]
    let mut found = false
    for j = 0; j < unique_labels.length(); j = j + 1 {
      if unique_labels[j] == label {
        found = true
        break
      }
    }
    if not(found) {
      unique_labels.push(label)
    }
  }
  assert_eq(unique_labels.length(), 2)
}

///|
test "AgglomerativeClustering - 全链接" {
  let x = @math.Matrix::new([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
  let model = AgglomerativeClustering::new(2, LinkageMethod::Complete)
  model.fit(x)
  let labels = model.get_labels()
  assert_eq(labels.length(), 3)

  // 前两个点应该在一个簇
  assert_eq(labels[0], labels[1])
}

///|
test "AgglomerativeClustering - 平均链接" {
  let x = @math.Matrix::new([[1.0, 1.0], [2.0, 2.0], [8.0, 8.0], [9.0, 9.0]])
  let model = AgglomerativeClustering::new(2, LinkageMethod::Average)
  let labels = model.fit_predict(x)
  assert_eq(labels.length(), 4)
}
