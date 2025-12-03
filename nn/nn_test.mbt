///|
/// NN 组件基础测试

///|
test "linear forward shape" {
  let lin = Linear::new(2, 3)
  let out = lin.forward(@math.Matrix::new([[1.0, 2.0]]))
  let shape = out.shape()
  assert_eq(shape.0, 1)
  assert_eq(shape.1, 3)
}

///|
test "activations basic" {
  let x = @math.Matrix::new([[-1.0, 0.0, 1.0]])
  let relu = activation_forward(x, act_relu())
  assert_eq(relu.get(0, 0), 0.0)
  let sig = activation_forward(x, act_sigmoid())
  assert_true(sig.get(0, 0) < 0.5)
  let sm = activation_forward(@math.Matrix::new([[1.0, 2.0]]), act_softmax())
  let s0 = sm.get(0, 0)
  let s1 = sm.get(0, 1)
  assert_true((s0 + s1 - 1.0).abs() < 1.0e-9)
  assert_true(s1 > s0)
}

///|
test "loss forward values" {
  let mse = mse_loss([1.0, 2.0], [1.0, 1.5])
  assert_true((mse - 0.125).abs() < 1.0e-9)
  let huber = huber_loss([1.0], [2.5], delta=1.0)
  assert_true((huber - 1.0).abs() < 1.0e-9)
  let bce = binary_cross_entropy([1.0, 0.0], [0.9, 0.1])
  assert_true(bce < 0.25)
  let logits = @math.Matrix::new([[2.0, 0.0]])
  let ce = cross_entropy([0], logits)
  assert_true(ce < 0.2)
}

///|
test "optimizers update" {
  let mut param = OptimParam::new([1.0, -1.0])
  param = param.with_grad([0.1, -0.2])
  let sgd = SGD::new(0.1, 2)
  sgd.step(param)
  assert_true((param.data[0] - 0.99).abs() < 1.0e-9)
  let mut adam_param = OptimParam::new([1.0, -1.0])
  adam_param = adam_param.with_grad([0.1, -0.1])
  let adam = Adam::new(0.1, 2)
  adam.step(adam_param)
  assert_true(adam_param.data[0] < 1.0)
}

///|
test "mlp forward logits" {
  let mlp = MLP::new(2, [3], 2, output_activation=act_softmax(), seed=1)
  let x = @math.Matrix::new([[0.0, 0.0], [1.0, 1.0]])
  let out = mlp.forward(x)
  let shape = out.shape()
  assert_eq(shape.0, 2)
  assert_eq(shape.1, 2)
  let probs = out.get_row(0)
  let sum = probs.get(0) + probs.get(1)
  assert_true((sum - 1.0).abs() < 1.0e-9)
}
