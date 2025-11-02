///|
/// 线性分类模型测试

///|
test "LogisticRegression - 二分类" {
  // 简单的线性可分数据
  let x = @math.Matrix::new([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 3.0],
    [6.0, 5.0],
    [7.0, 8.0],
    [8.0, 8.0],
  ])
  let y = @math.Vector::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
  let model = LogisticRegression::new(0.1, 1000, 1.0e-4)
  model.fit(x, y)
  assert_true(model.is_fitted)
  let score = model.score(x, y)
  // println(score)
  assert_true(score > 0.8) // 准确率 > 80%
}

///|
test "LogisticRegression - 预测概率" {
  let x = @math.Matrix::new([[1.0, 1.0], [2.0, 2.0], [8.0, 8.0], [9.0, 9.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let model = LogisticRegression::new(0.1, 1000, 1.0e-4)
  model.fit(x, y)
  let probas = model.predict_proba(x)
  // probas.get(2) |> println
  // 检查概率在 [0, 1] 之间
  for i = 0; i < probas.size(); i = i + 1 {
    assert_true(probas.get(i) >= 0.0)
    assert_true(probas.get(i) <= 1.0)
  }

  // 第一组应该更接近0，第二组应该更接近1
  assert_true(probas.get(0) < 0.5)
  assert_true(probas.get(2) > 0.5)
}

///|
test "Perceptron - 感知机" {
  // 线性可分数据
  let x = @math.Matrix::new([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
  // AND 问题
  let y = @math.Vector::new([0.0, 0.0, 0.0, 1.0])
  let model = Perceptron::new(0.1, 100)
  model.fit(x, y)
  assert_true(model.is_fitted)
  let y_pred = model.predict(x)

  // 检查前3个样本预测为0
  assert_eq(y_pred.get(0), 0.0)
  assert_eq(y_pred.get(1), 0.0)
  assert_eq(y_pred.get(2), 0.0)

  // 最后一个预测为1
  assert_eq(y_pred.get(3), 1.0)
}

///|
test "SGDClassifier - Hinge Loss" {
  let x = @math.Matrix::new([[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let model = SGDClassifier::new(0.01, 100, 0.001, LossFunction::Hinge)
  model.fit(x, y)
  assert_true(model.is_fitted)
  let score = model.score(x, y)
  assert_true(score >= 0.5) // 至少50%准确率
}

///|
test "SGDClassifier - Log Loss" {
  let x = @math.Matrix::new([[1.0, 1.0], [2.0, 2.0], [6.0, 6.0], [7.0, 7.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let model = SGDClassifier::new(0.1, 200, 0.001, LossFunction::Log)
  model.fit(x, y)
  let score = model.score(x, y)
  assert_true(score > 0.7)
}

///|
test "SGDClassifier - Perceptron Loss" {
  let x = @math.Matrix::new([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
  let y = @math.Vector::new([0.0, 0.0, 0.0, 1.0])
  let model = SGDClassifier::new(0.1, 100, 0.0, LossFunction::Perceptron)
  model.fit(x, y)
  let score = model.score(x, y)
  assert_true(score >= 0.75)
}
