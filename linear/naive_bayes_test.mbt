///|
/// 朴素贝叶斯测试

///|
test "GaussianNB classifies simple data" {
  let x = @math.Matrix::new([[1.0, 1.0], [1.1, 1.0], [4.0, 4.0], [4.1, 4.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let model = GaussianNB::new()
  model.fit(x, y)
  let acc = model.score(x, y)
  assert_true(acc == 1.0)
}

///|
test "MultinomialNB prefers frequent counts" {
  let x = @math.Matrix::new([[2.0, 1.0], [3.0, 1.0], [0.0, 5.0], [0.0, 4.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let model = MultinomialNB::new(alpha=1.0)
  model.fit(x, y)
  let acc = model.score(x, y)
  assert_true(acc == 1.0)
}

///|
test "BernoulliNB works on binary features" {
  let x = @math.Matrix::new([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let model = BernoulliNB::new(alpha=1.0)
  model.fit(x, y)
  let acc = model.score(x, y)
  assert_true(acc == 1.0)
}
