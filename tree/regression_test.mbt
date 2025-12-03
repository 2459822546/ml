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

///|
test "ExtraTreesRegressor basic" {
  let x = @math.Matrix::new([[0.0], [1.0], [2.0], [3.0]])
  let y = @math.Vector::new([0.0, 1.0, 2.0, 3.0])
  let et = ExtraTreesRegressor::new(n_estimators=4, max_depth=3, seed=9)
  et.fit(x, y)
  let pred = et.predict(x)
  assert_true((pred.get(0) - 0.0).abs() < 1.0e-6)
}

///|
test "AdaBoostRegressor smoothing" {
  let x = @math.Matrix::new([[0.0], [1.0], [2.0], [3.0]])
  let y = @math.Vector::new([0.0, 1.0, 2.0, 3.0])
  let ada = AdaBoostRegressor::new(
    n_estimators=5,
    max_depth=2,
    learning_rate=0.5,
  )
  ada.fit(x, y)
  let pred = ada.predict(x)
  assert_true((pred.get(2) - 2.0).abs() < 0.6)
}

///|
test "GradientBoostingRegressor residual updates" {
  let x = @math.Matrix::new([[0.0], [1.0], [2.0], [3.0]])
  let y = @math.Vector::new([0.0, 1.0, 2.0, 3.0])
  let gb = GradientBoostingRegressor::new(
    n_estimators=8,
    max_depth=2,
    learning_rate=0.3,
  )
  gb.fit(x, y)
  let pred = gb.predict(x)
  let mut err = 0.0
  for i = 0; i < y.size(); i = i + 1 {
    let diff = y.get(i) - pred.get(i)
    err = err + diff * diff
  }
  let mse_val = err / y.size().to_double()
  assert_true(mse_val < 0.2)
}
