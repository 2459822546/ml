///|
/// KNN 分类与回归测试

///|
test "knn classifier simple" {
  let x = @math.Matrix::new([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let clf = KNeighborsClassifier::new(k=3)
  clf.fit(x, y)
  let pred = clf.predict(@math.Matrix::new([[0.1, 0.1], [0.9, 0.9]]))
  assert_eq(pred.get(0), 0.0)
  assert_eq(pred.get(1), 1.0)
}

///|
test "knn regressor mean" {
  let x = @math.Matrix::new([[0.0], [1.0], [2.0]])
  let y = @math.Vector::new([0.0, 1.0, 2.0])
  let reg = KNeighborsRegressor::new(k=2)
  reg.fit(x, y)
  let pred = reg.predict(@math.Matrix::new([[1.0]]))
  assert_true((pred.get(0) - 0.5).abs() < 1.0e-9)
}
