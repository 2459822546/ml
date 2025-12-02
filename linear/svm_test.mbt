///|
/// 线性 SVM 测试

///|
test "LinearSVM separates simple margin data" {
  // 标签为 +1/-1
  let x = @math.Matrix::new([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [5.0, 5.0]])
  let y = @math.Vector::new([1.0, 1.0, -1.0, -1.0])
  let model = LinearSVM::new(c=1.0, learning_rate=0.2, max_iter=500)
  model.fit(x, y)
  let acc = model.score(x, y)
  assert_true(acc >= 0.75)
}
