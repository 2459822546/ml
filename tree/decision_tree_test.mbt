///|
/// 决策树与随机森林测试

///|
test "DecisionTreeClassifier fits XOR" {
  let x = @math.Matrix::new([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
  let y = @math.Vector::new([0.0, 1.0, 1.0, 0.0])
  let tree = DecisionTreeClassifier::new(max_depth=3)
  tree.fit(x, y)
  let acc = tree.score(x, y)
  assert_true(acc == 1.0)
}

///|
test "RandomForestClassifier majority vote" {
  let x = @math.Matrix::new([
    [1.0, 1.0],
    [1.2, 1.1],
    [3.0, 3.0],
    [3.1, 3.2],
    [5.0, 0.0],
    [5.1, 0.1],
  ])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
  let rf = RandomForestClassifier::new(
    n_estimators=1,
    max_depth=4,
    max_features=2,
  )
  rf.fit(x, y)
  let preds = rf.predict(x)
  assert_eq(preds.size(), y.size())
  let imp = rf.feature_importances()
  assert_eq(imp.size(), 2)
  assert_true(imp.get(0) >= 0.0)
}
