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

///|
test "dropout backward uses mask" {
  let x = AutoTensor::new(@math.Tensor::new([1.0, 1.0, 1.0, 1.0], [2, 2]))
  let (y, _seed) = x.dropout(0.5, 7, train=true)
  let loss = y.sum()
  loss.backward()
  let grad = x.grad()
  let out = y.value()
  assert_eq(grad.get([0, 0]), out.get([0, 0]))
  assert_eq(grad.get([0, 1]), out.get([0, 1]))
  assert_eq(grad.get([1, 0]), out.get([1, 0]))
  assert_eq(grad.get([1, 1]), out.get([1, 1]))
}

///|
fn batchnorm_loss(x : @math.Tensor) -> Double {
  let bn = BatchNorm1d::new(2)
  let out = bn.forward(x.to_matrix(), train=true)
  let (n, m) = out.shape()
  let mut sum = 0.0
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      sum = sum + out.get(i, j)
    }
  }
  sum
}

///|
test "batchnorm autograd numeric grad" {
  let base = [1.0, 2.0, 3.0, 4.0]
  let shape = [2, 2]
  let x = AutoTensor::new(@math.Tensor::new(base, shape))
  let bn = BatchNorm1d::new(2)
  let (y, _ctx) = module_forward(module_batchnorm1d(bn), x, train=true)
  let loss = y.sum()
  loss.backward()
  let grad = x.grad()
  let eps = 1.0e-5
  for i = 0; i < 2; i = i + 1 {
    for j = 0; j < 2; j = j + 1 {
      let idx = i * 2 + j
      let plus = Array::make(4, 0.0)
      let minus = Array::make(4, 0.0)
      for k = 0; k < 4; k = k + 1 {
        plus[k] = base[k]
        minus[k] = base[k]
      }
      plus[idx] = plus[idx] + eps
      minus[idx] = minus[idx] - eps
      let loss_plus = batchnorm_loss(@math.Tensor::new(plus, shape))
      let loss_minus = batchnorm_loss(@math.Tensor::new(minus, shape))
      let num_grad = (loss_plus - loss_minus) / (2.0 * eps)
      let diff = (grad.get([i, j]) - num_grad).abs()
      assert_true(diff < 1.0e-3)
    }
  }
}

///|
fn layernorm_loss(x : @math.Tensor) -> Double {
  let ln = LayerNorm::new(2)
  let out = ln.forward(x.to_matrix())
  let (n, m) = out.shape()
  let mut sum = 0.0
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      sum = sum + out.get(i, j)
    }
  }
  sum
}

///|
test "layernorm autograd numeric grad" {
  let base = [1.0, -1.0, 2.0, 0.5]
  let shape = [2, 2]
  let x = AutoTensor::new(@math.Tensor::new(base, shape))
  let ln = LayerNorm::new(2)
  let (y, _ctx) = module_forward(module_layernorm(ln), x, train=true)
  let loss = y.sum()
  loss.backward()
  let grad = x.grad()
  let eps = 1.0e-5
  for i = 0; i < 2; i = i + 1 {
    for j = 0; j < 2; j = j + 1 {
      let idx = i * 2 + j
      let plus = Array::make(4, 0.0)
      let minus = Array::make(4, 0.0)
      for k = 0; k < 4; k = k + 1 {
        plus[k] = base[k]
        minus[k] = base[k]
      }
      plus[idx] = plus[idx] + eps
      minus[idx] = minus[idx] - eps
      let loss_plus = layernorm_loss(@math.Tensor::new(plus, shape))
      let loss_minus = layernorm_loss(@math.Tensor::new(minus, shape))
      let num_grad = (loss_plus - loss_minus) / (2.0 * eps)
      let diff = (grad.get([i, j]) - num_grad).abs()
      assert_true(diff < 1.0e-3)
    }
  }
}
