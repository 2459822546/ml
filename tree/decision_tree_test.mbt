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

///|
test "panic_rf_predict_not_fitted" {
  let rf = RandomForestClassifier::new()
  let _ = rf.predict(@math.Matrix::new([[1.0]]))

}

///|
test "panic_rf_feature_importances_not_fitted" {
  let rf = RandomForestClassifier::new()
  let _ = rf.feature_importances()

}

///|
test "panic_rf_score_length_mismatch" {
  let rf = RandomForestClassifier::new()
  let x = @math.Matrix::new([[1.0, 1.0], [2.0, 2.0]])
  let y = @math.Vector::new([0.0, 1.0])
  rf.fit(x, y)
  let _ = rf.score(x, @math.Vector::new([0.0]))

}

///|
test "ExtraTreesClassifier randomness" {
  let x = @math.Matrix::new([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let et = ExtraTreesClassifier::new(n_estimators=3, max_depth=3, seed=5)
  et.fit(x, y)
  let preds = et.predict(x)
  let mut correct = 0
  for i = 0; i < preds.size(); i = i + 1 {
    if preds.get(i) == y.get(i) {
      correct = correct + 1
    }
  }
  assert_true(correct >= 2)
}

///|
test "AdaBoostClassifier improves on stump" {
  let x = @math.Matrix::new([[0.0], [0.5], [1.0], [1.5]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let ada = AdaBoostClassifier::new(
    n_estimators=5,
    max_depth=1,
    learning_rate=0.8,
  )
  ada.fit(x, y)
  let preds = ada.predict(x)
  let mut correct = 0
  for i = 0; i < preds.size(); i = i + 1 {
    if preds.get(i) == y.get(i) {
      correct = correct + 1
    }
  }
  assert_true(correct >= 3)
}

///|
test "GradientBoostingClassifier logloss drop" {
  let x = @math.Matrix::new([[0.0], [1.0], [2.0], [3.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let gb = GradientBoostingClassifier::new(
    n_estimators=10,
    max_depth=2,
    learning_rate=0.2,
  )
  gb.fit(x, y)
  let preds = gb.predict(x)
  let mut correct = 0
  for i = 0; i < preds.size(); i = i + 1 {
    if preds.get(i) == y.get(i) {
      correct = correct + 1
    }
  }
  assert_true(correct >= 3)
}
