///|
/// 线性回归模型测试

///|
test "LinearRegression - 简单线性回归" {
  // y = 2x + 3
  let x = @math.Matrix::new([[1.0], [2.0], [3.0], [4.0], [5.0]])
  let y = @math.Vector::new([5.0, 7.0, 9.0, 11.0, 13.0])
  let model = LinearRegression::new()
  model.fit(x, y)

  // 检查权重和截距
  assert_true((model.weights.get(0) - 2.0).abs() < 0.1)
  assert_true((model.intercept - 3.0).abs() < 0.1)

  // 测试预测
  let x_test = @math.Matrix::new([[6.0]])
  let y_pred = model.predict(x_test)
  assert_true((y_pred.get(0) - 15.0).abs() < 0.5)
}

///|
test "LinearRegression - 多元线性回归" {
  // y = 1*x1 + 2*x2 + 3
  let x = @math.Matrix::new([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
  let y = @math.Vector::new([6.0, 9.0, 12.0, 15.0])
  let model = LinearRegression::new()
  model.fit(x, y)
  let score = model.score(x, y)
  score |> println
  assert_true(score > 0.95) // R² > 0.95
}

///|
test "Ridge - 岭回归" {
  let x = @math.Matrix::new([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
  let y = @math.Vector::new([3.0, 5.0, 7.0, 9.0])
  let model = Ridge::new(1.0)
  model.fit(x, y)
  let score = model.score(x, y)
  assert_true(score > 0.8)
}

///|
test "Lasso - Lasso回归" {
  let x = @math.Matrix::new([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4]])
  let y = @math.Vector::new([1.0, 2.0, 3.0, 4.0])
  let model = Lasso::new(0.1, 1000, 1.0e-4)
  model.fit(x, y)
  assert_true(model.is_fitted)
  let y_pred = model.predict(x)
  // 检查预测不会完全偏离
  for i = 0; i < y.size(); i = i + 1 {
    assert_true((y_pred.get(i) - y.get(i)).abs() < 2.0)
  }
}

///|
test "ElasticNet - 弹性网络" {
  let x = @math.Matrix::new([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
  let y = @math.Vector::new([2.0, 3.5, 5.0, 6.5])
  let model = ElasticNet::new(0.1, 0.5, 1000, 1.0e-4)
  model.fit(x, y)
  assert_true(model.is_fitted)
  let y_pred = model.predict(x)
  for i = 0; i < y.size(); i = i + 1 {
    assert_true((y_pred.get(i) - y.get(i)).abs() < 2.0)
  }
}
