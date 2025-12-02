///|
/// One-vs-Rest 多分类测试

///|
test "OvR with three classes" {
  // 三类可分数据
  let x = @math.Matrix::new([
    [1.0, 0.0],
    [2.0, 0.0],
    [0.0, 1.0],
    [0.0, 2.0],
    [1.0, 1.0],
    [2.0, 2.0],
  ])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
  let model = OneVsRestClassifier::new(
    learning_rate=0.5,
    max_iter=200,
    tol=1.0e-4,
  )
  model.fit(x, y)
  let pred = model.predict(x)
  for i = 0; i < y.size(); i = i + 1 {
    assert_eq(pred.get(i), y.get(i))
  }
}
