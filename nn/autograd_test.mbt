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

///|
test "matmul add_bias sum gradients" {
  let graph = ComputationGraph::new()
  let x = graph.add_input(
    @math.Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
  )
  let w = graph.add_input(
    @math.Tensor::new([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2]),
  )
  let b = graph.add_input(@math.Tensor::new([0.5, -0.5], [2]))
  let y = graph.matmul(x, w)
  let yb = graph.add_bias(y, b)
  let loss = graph.sum(yb)
  graph.backward(loss)
  let gw = graph.grad_of(w)
  assert_true((gw.get([0, 0]) - 5.0).abs() < 1.0e-9)
  assert_true((gw.get([0, 1]) - 5.0).abs() < 1.0e-9)
  assert_true((gw.get([1, 0]) - 7.0).abs() < 1.0e-9)
  assert_true((gw.get([2, 1]) - 9.0).abs() < 1.0e-9)
  let gb = graph.grad_of(b)
  assert_eq(gb.get([0]), 2.0)
  assert_eq(gb.get([1]), 2.0)
}

///|
test "autotensor style gradients" {
  let x = AutoTensor::new(@math.Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2]))
  let w = x.from_other(@math.Tensor::new([1.0, 0.0, 0.0, 1.0], [2, 2]))
  let b = x.from_other(@math.Tensor::new([0.5, -0.5], [2]))
  let y = x.matmul(w).add_bias(b)
  let loss = y.sum()
  loss.backward()
  let gw = w.grad()
  assert_true((gw.get([0, 0]) - 4.0).abs() < 1.0e-9)
  assert_true((gw.get([1, 1]) - 6.0).abs() < 1.0e-9)
  let gb = b.grad()
  assert_eq(gb.get([0]), 2.0)
  assert_eq(gb.get([1]), 2.0)
}
