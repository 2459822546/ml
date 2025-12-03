///|
/// 计算图反向传播测试

///|
test "mul add relu gradients" {
  let graph = ComputationGraph::new()
  let a = graph.add_input(@math.Tensor::from_vector(@math.Vector::new([1.0])))
  let b = graph.add_input(@math.Tensor::from_vector(@math.Vector::new([2.0])))
  let c = graph.mul(a, b) // 2
  let d = graph.add(c, b) // 4
  let e = graph.relu(d)
  graph.backward(e)
  let grad_a = graph.grad_of(a)
  let grad_b = graph.grad_of(b)
  assert_true((grad_a.get([0]) - 2.0).abs() < 1.0e-9)
  assert_true((grad_b.get([0]) - 2.0).abs() < 1.0e-9)
}
