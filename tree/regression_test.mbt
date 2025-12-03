///|
/// 决策树回归与随机森林回归测试

///|
test "DecisionTreeRegressor fits simple line" {
  let x = @math.Matrix::new([[1.0], [2.0], [3.0], [4.0]])
  let y = @math.Vector::new([2.0, 4.0, 6.0, 8.0])
  let tree = DecisionTreeRegressor::new(max_depth=3)
  tree.fit(x, y)
  let pred = tree.predict(x)
  for i = 0; i < y.size(); i = i + 1 {
    assert_true((pred.get(i) - y.get(i)).abs() < 1.0e-9)
  }
}

///|
test "RandomForestRegressor averages predictions" {
  let x = @math.Matrix::new([[1.0], [2.0], [3.0], [4.0]])
  let y = @math.Vector::new([1.0, 2.0, 3.0, 4.0])
  let rf = RandomForestRegressor::new(
    n_estimators=3,
    max_depth=3,
    max_features=1,
  )
  rf.fit(x, y)
  let pred = rf.predict(x)
  for i = 0; i < y.size(); i = i + 1 {
    assert_true((pred.get(i) - y.get(i)).abs() < 1.0e-6)
  }
  let imp = rf.feature_importances()
  assert_eq(imp.size(), 1)
}
