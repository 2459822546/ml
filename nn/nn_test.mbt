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
test "layers basic" {
  let x = @math.Matrix::new([[1.0, 1.0], [1.0, 1.0]])
  let drop = Dropout::new(0.0, seed=1)
  let dropped = drop.forward(x, train=true)
  assert_eq(dropped.get(0, 0), 1.0)
  let drop2 = Dropout::new(0.5, seed=7)
  let dropped2 = drop2.forward(x, train=true)
  let v = dropped2.get(0, 0)
  assert_true(v == 0.0 || (v - 2.0).abs() < 1.0e-9)
  let bn = BatchNorm1d::new(2)
  let bn_out = bn.forward(@math.Matrix::new([[1.0, 2.0], [3.0, 4.0]]))
  assert_true((bn_out.get(0, 0) + 1.0).abs() < 1.0e-4)
  assert_true((bn_out.get(1, 1) - 1.0).abs() < 1.0e-4)
  let ln = LayerNorm::new(2)
  let ln_out = ln.forward(@math.Matrix::new([[1.0, 2.0], [3.0, 4.0]]))
  assert_true((ln_out.get(0, 0) + 1.0).abs() < 1.0e-4)
  assert_true((ln_out.get(1, 1) - 1.0).abs() < 1.0e-4)
  let emb = Embedding::new(3, 2, seed=9)
  let emb_out = emb.forward([0, 2])
  assert_eq(emb_out.shape(), (2, 2))
  assert_eq(emb_out.get(0, 0), emb.weight[0][0])
  assert_eq(emb_out.get(1, 1), emb.weight[2][1])
}

///|
test "module parameter autograd" {
  let x = AutoTensor::new(@math.Tensor::new([1.0, 2.0], [1, 2]))
  let linear = LinearModule::new(2, 1, seed=1)
  linear.set_params(
    @math.Tensor::new([1.0, 2.0], [2, 1]),
    @math.Tensor::new([0.5], [1]),
  )
  let (y, ctx) = linear.forward(x)
  let loss = y.sum()
  loss.backward()
  linear.sync_grads(ctx)
  let params = linear.parameters()
  let gw = params[0].get_grad()
  let gb = params[1].get_grad()
  assert_true((gw.get([0, 0]) - 1.0).abs() < 1.0e-9)
  assert_true((gw.get([1, 0]) - 2.0).abs() < 1.0e-9)
  assert_eq(gb.get([0]), 1.0)
}

///|
test "sequential module params and forward" {
  let x = AutoTensor::new(@math.Tensor::new([1.0, 2.0], [1, 2]))
  let linear = LinearModule::new(2, 2, seed=1)
  linear.set_params(
    @math.Tensor::new([1.0, 0.0, 0.0, 1.0], [2, 2]),
    @math.Tensor::new([0.0, 0.0], [2]),
  )
  let seq = Sequential::new([module_linear(linear), module_relu()])
  let (y, ctx) = seq.forward(x)
  let loss = y.sum()
  loss.backward()
  seq.sync_grads(ctx)
  let params = seq.parameters()
  assert_eq(params.length(), 2)
  let gw = params[0].get_grad()
  assert_true((gw.get([0, 0]) - 1.0).abs() < 1.0e-9)
}

///|
test "module train eval with dropout" {
  let x = AutoTensor::new(@math.Tensor::new([1.0, 1.0, 1.0, 1.0], [2, 2]))
  let drop = Dropout::new(0.5, seed=3)
  let seq = Sequential::new([module_dropout(drop)])
  seq.eval()
  let (y, _ctx) = seq.forward(x)
  let out = y.value()
  assert_eq(out.get([0, 0]), 1.0)
  assert_eq(out.get([1, 1]), 1.0)
}

///|
test "modulelist nested parameters" {
  let linear = LinearModule::new(2, 1, seed=1)
  let inner = ModuleList::new([module_linear(linear)])
  let outer = ModuleList::new([module_modulelist(inner), module_relu()])
  let params = outer.parameters()
  assert_eq(params.length(), 2)
}

///|
test "batchnorm layernorm parameter grads" {
  let x = AutoTensor::new(@math.Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2]))
  let bn = BatchNorm1dModule::new(2)
  let (y, ctx) = bn.forward(x, train=true)
  let loss = y.sum()
  loss.backward()
  bn.sync_grads(ctx)
  let bn_params = bn.parameters()
  let bn_gamma_grad = bn_params[0].get_grad()
  let bn_beta_grad = bn_params[1].get_grad()
  assert_true(bn_gamma_grad.get([0]).abs() < 1.0e-6)
  assert_true((bn_beta_grad.get([0]) - 2.0).abs() < 1.0e-6)
  let ln = LayerNormModule::new(2)
  let (y2, ctx2) = ln.forward(x)
  let loss2 = y2.sum()
  loss2.backward()
  ln.sync_grads(ctx2)
  let ln_params = ln.parameters()
  let ln_gamma_grad = ln_params[0].get_grad()
  let ln_beta_grad = ln_params[1].get_grad()
  assert_true((ln_gamma_grad.get([0]) + 2.0).abs() < 1.0e-4)
  assert_true((ln_gamma_grad.get([1]) - 2.0).abs() < 1.0e-4)
  assert_true((ln_beta_grad.get([0]) - 2.0).abs() < 1.0e-6)
}

///|
test "embedding module grads" {
  let emb = EmbeddingModule::new(3, 2, seed=1)
  let x = AutoTensor::new(@math.Tensor::new([0.0, 2.0], [2]))
  let (y, ctx) = emb.forward(x)
  let loss = y.sum()
  loss.backward()
  emb.sync_grads(ctx)
  let grad = emb.parameters()[0].get_grad()
  assert_true((grad.get([0, 0]) - 1.0).abs() < 1.0e-9)
  assert_true((grad.get([0, 1]) - 1.0).abs() < 1.0e-9)
  assert_true((grad.get([2, 0]) - 1.0).abs() < 1.0e-9)
  assert_true((grad.get([2, 1]) - 1.0).abs() < 1.0e-9)
  assert_eq(grad.get([1, 0]), 0.0)
  assert_eq(grad.get([1, 1]), 0.0)
}

///|
test "state_dict load_state_dict basic" {
  let linear = LinearModule::new(2, 2, seed=2)
  linear.set_params(
    @math.Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2]),
    @math.Tensor::new([0.5, -0.5], [2]),
  )
  let bn = BatchNorm1d::new(2)
  let ln = LayerNorm::new(2)
  let seq = Sequential::new([
    module_linear(linear),
    module_batchnorm1d(bn),
    module_layernorm(ln),
  ])
  let x = AutoTensor::new(@math.Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2]))
  let _ = seq.forward(x)
  let state = seq.state_dict()
  let params = seq.parameters()
  params[0].set_data(@math.Tensor::zeros([2, 2]))
  params[1].set_data(@math.Tensor::zeros([2]))
  seq.load_state_dict(state)
  let params_after = seq.parameters()
  assert_eq(params_after[0].get_data().get([0, 0]), 1.0)
  assert_eq(params_after[1].get_data().get([0]), 0.5)
}

///|
test "embedding state_dict load" {
  let emb = EmbeddingModule::new(3, 2, seed=1)
  let state = module_state_dict(module_embedding_module(emb))
  let params = emb.parameters()
  let w0 = params[0].get_data().get([0, 0])
  params[0].set_data(@math.Tensor::zeros([3, 2]))
  module_load_state_dict(module_embedding_module(emb), state)
  let w1 = emb.parameters()[0].get_data().get([0, 0])
  assert_eq(w1, w0)
}

///|
test "conv2d forward and grads" {
  let conv = Conv2dModule::new(1, 1, 1, stride=1, padding=0, seed=1)
  conv.set_params(
    @math.Tensor::new([2.0], [1, 1, 1, 1]),
    @math.Tensor::new([0.5], [1]),
  )
  let x = AutoTensor::new(@math.Tensor::new([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]))
  let (y, ctx) = conv.forward(x)
  let out = y.value()
  assert_eq(out.get([0, 0, 0, 0]), 2.5)
  assert_eq(out.get([0, 0, 1, 1]), 8.5)
  let loss = y.sum()
  loss.backward()
  conv.sync_grads(ctx)
  let params = conv.parameters()
  let grad_w = params[0].get_grad()
  let grad_b = params[1].get_grad()
  assert_true((grad_w.get([0, 0, 0, 0]) - 10.0).abs() < 1.0e-9)
  assert_true((grad_b.get([0]) - 4.0).abs() < 1.0e-9)
  let gx = x.grad()
  assert_true((gx.get([0, 0, 0, 0]) - 2.0).abs() < 1.0e-9)
  assert_true((gx.get([0, 0, 1, 1]) - 2.0).abs() < 1.0e-9)
}

///|
test "pooling and flatten2d" {
  let x = AutoTensor::new(@math.Tensor::new([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]))
  let maxp = MaxPool2dModule::new(2)
  let avgp = AvgPool2dModule::new(2)
  let flat = Flatten2dModule::new()
  let y_max = maxp.forward(x)
  let y_avg = avgp.forward(x)
  let y_flat = flat.forward(x)
  assert_eq(y_max.value().get([0, 0, 0, 0]), 4.0)
  assert_true((y_avg.value().get([0, 0, 0, 0]) - 2.5).abs() < 1.0e-9)
  let shape = y_flat.value().get_shape()
  assert_eq(shape[0], 1)
  assert_eq(shape[1], 4)
  let loss = y_max.sum()
  loss.backward()
  let gx = x.grad()
  assert_eq(gx.get([0, 0, 1, 1]), 1.0)
  assert_eq(gx.get([0, 0, 0, 0]), 0.0)
}

///|
test "autotensor operators" {
  let a = AutoTensor::new(@math.Tensor::from_vector(@math.Vector::new([2.0])))
  let b = a.from_other(@math.Tensor::from_vector(@math.Vector::new([3.0])))
  let c = a + b
  let d = c * b
  let loss = d.sum()
  loss.backward()
  let ga = a.grad()
  let gb = b.grad()
  assert_true((ga.get([0]) - 3.0).abs() < 1.0e-9)
  assert_true((gb.get([0]) - 8.0).abs() < 1.0e-9)
}

///|
test "activations basic" {
  let x = @math.Matrix::new([[-1.0, 0.0, 1.0]])
  let relu = activation_forward(x, act_relu())
  assert_eq(relu.get(0, 0), 0.0)
  let sig = activation_forward(x, act_sigmoid())
  assert_true(sig.get(0, 0) < 0.5)
  let th = activation_forward(x, act_tanh())
  assert_true(th.get(0, 2) < 1.0 && th.get(0, 0) < 0.0)
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
  let mae = mae_loss([1.0, 2.0], [2.0, 1.0])
  assert_true((mae - 1.0).abs() < 1.0e-9)
  let sl1 = smooth_l1_loss([1.0], [3.0], beta=1.0)
  assert_true((sl1 - 1.5).abs() < 1.0e-9)
  let bce = binary_cross_entropy([1.0, 0.0], [0.9, 0.1])
  assert_true(bce < 0.25)
  let mbce = multilabel_binary_cross_entropy([[1.0, 0.0], [0.0, 1.0]], [
    [0.9, 0.1],
    [0.2, 0.8],
  ])
  assert_true(mbce < 0.3)
  let focal = focal_loss_binary([1.0, 0.0], [0.9, 0.1])
  assert_true(focal < 0.05)
  let logits = @math.Matrix::new([[2.0, 0.0]])
  let ce = cross_entropy([0], logits)
  assert_true(ce < 0.2)
  let ls = label_smoothing_cross_entropy([0], logits, smoothing=0.1)
  assert_true(ls > ce)
  let kl = kl_divergence([0.7, 0.3], [0.6, 0.4])
  assert_true(kl > 0.0)
  let hinge = hinge_loss([1.0, -1.0], [2.0, -0.5])
  assert_true((hinge - 0.25).abs() < 1.0e-9)
  let ce2 = softmax_cross_entropy([0], logits)
  assert_true((ce2 - ce).abs() < 1.0e-9)
  let (ce3, probs) = softmax_cross_entropy_with_probs([0], logits)
  assert_true((ce3 - ce).abs() < 1.0e-9)
  let sum_probs = probs.get(0, 0) + probs.get(0, 1)
  assert_true((sum_probs - 1.0).abs() < 1.0e-9)
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
  let mut adamw_param = OptimParam::new([1.0, -1.0])
  adamw_param = adamw_param.with_grad([0.1, -0.1])
  let adamw = AdamW::new(0.1, 2, weight_decay=0.1)
  adamw.step(adamw_param)
  assert_true(adamw_param.data[0] < 1.0)
  let mut rms_param = OptimParam::new([1.0, -1.0])
  rms_param = rms_param.with_grad([0.1, -0.1])
  let rms = RMSProp::new(0.1, 2)
  rms.step(rms_param)
  assert_true(rms_param.data[0] < 1.0)
  let mut ada_param = OptimParam::new([1.0, -1.0])
  ada_param = ada_param.with_grad([0.1, -0.1])
  let adagrad = Adagrad::new(0.1, 2)
  adagrad.step(ada_param)
  assert_true(ada_param.data[0] < 1.0)
}

///|
test "optimizer wrapper and schedules" {
  let mut param = OptimParam::new([1.0])
  param = param.with_grad([0.1])
  let opt = optimizer_sgd(SGD::new(0.1, 1))
  let _ = optimizer_step(opt, param)
  assert_true(param.data[0] < 1.0)
  let opt2 = optimizer_set_lr(opt, 0.05)
  assert_true((optimizer_get_lr(opt2) - 0.05).abs() < 1.0e-9)
  let cos0 = cosine_decay(0.1, 0.0, 0, 10)
  let cos_end = cosine_decay(0.1, 0.0, 10, 10)
  assert_true((cos0 - 0.1).abs() < 1.0e-9)
  assert_true((cos_end - 0.0).abs() < 1.0e-9)
  let warm0 = linear_warmup(0.1, 5, 0)
  let warm3 = linear_warmup(0.1, 5, 3)
  assert_true((warm0 - 0.0).abs() < 1.0e-9)
  assert_true((warm3 - 0.06).abs() < 1.0e-9)
  let wc = warmup_cosine(0.1, 0.0, 2, 2, 6)
  assert_true((wc - 0.1).abs() < 1.0e-9)
}

///|
test "optimizer params step" {
  let p = Parameter::new(@math.Tensor::new([1.0], [1]))
  p.set_grad(@math.Tensor::new([0.1], [1]))
  let opt = SGDParams::new(0.1, [p])
  opt.step([p])
  let v = p.get_data().get([0])
  assert_true((v - 0.99).abs() < 1.0e-9)
}

///|
test "optimizer params step seq" {
  let linear = LinearModule::new(2, 1, seed=1)
  linear.set_params(
    @math.Tensor::new([1.0, 2.0], [2, 1]),
    @math.Tensor::new([0.0], [1]),
  )
  let seq = Sequential::new([module_linear(linear)])
  let x = AutoTensor::new(@math.Tensor::new([1.0, 2.0], [1, 2]))
  let (y, ctx) = seq.forward(x)
  let loss = y.sum()
  loss.backward()
  seq.sync_grads(ctx)
  let params_before = seq.parameters()
  let w_before = params_before[0].get_data().get([0, 0])
  let opt = SGDParams::new(0.1, params_before)
  opt.step_seq(seq)
  let params_after = seq.parameters()
  let w_after = params_after[0].get_data().get([0, 0])
  assert_true(w_after < w_before)
}

///|
test "named parameters and buffers" {
  let linear = LinearModule::new(2, 1, seed=1)
  let bn = BatchNorm1d::new(1)
  let seq = Sequential::new([module_linear(linear), module_batchnorm1d(bn)])
  let params = seq.named_parameters()
  match params.get("0.weight") {
    Some(_) => ()
    None => abort("缺少 0.weight")
  }
  match params.get("1.gamma") {
    Some(_) => ()
    None => abort("缺少 1.gamma")
  }
  let buffers = seq.named_buffers()
  match buffers.get("1.running_mean") {
    Some(_) => ()
    None => abort("缺少 1.running_mean")
  }
  match buffers.get("1.running_var") {
    Some(_) => ()
    None => abort("缺少 1.running_var")
  }
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

///|
test "training utils" {
  let mut p = OptimParam::new([1.0, 2.0])
  p = p.with_grad([3.0, -4.0])
  clip_grad_value([p], 2.0)
  assert_eq(p.grad[0], 2.0)
  assert_eq(p.grad[1], -2.0)
  let before = clip_grad_norm([p], 2.5)
  assert_true((before - 2.828427).abs() < 1.0e-5)
  let expected = 2.5 / (before + 1.0e-6)
  assert_true((p.grad[0] - 2.0 * expected).abs() < 1.0e-6)
  let es = early_stopping_min(2, min_delta=0.01)
  assert_true(not(es.update(1.0)))
  assert_true(not(es.update(0.9)))
  assert_true(not(es.update(0.91)))
  assert_true(es.update(0.92))
  let es_max = early_stopping_max(1)
  assert_true(not(es_max.update(1.0)))
  assert_true(es_max.update(0.9))
}

///|
test "mlp train loop with dataloader" {
  let features = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
  let labels = [0.0, 1.0, 1.0, 0.0]
  let ds = @data.Dataset::new(features, labels)
  let loader = @data.DataLoader::new(ds, batch_size=2, shuffle=false)
  let mlp = MLP::new(2, [2], 2, output_activation=act_softmax(), seed=3)
  let trained = mlp_train_loop(mlp, loader, 1, fn(
    m,
    batch_x,
    _batch_y,
    _epoch,
    _batch_idx,
  ) {
    let x = @math.Matrix::new(batch_x)
    let _ = m.forward(x)
    m
  })
  let out = trained.forward(@math.Matrix::new([[0.0, 0.0]]))
  assert_eq(out.shape().1, 2)
}

///|
test "tensor forward compat" {
  let lin = Linear::new(2, 2)
  let w = [[1.0, 0.0], [0.0, 1.0]]
  let b = [0.0, 0.0]
  lin.set_params(w, b)
  let t = @math.Tensor::from_matrix(@math.Matrix::new([[1.0, 2.0]]))
  let out = lin.forward_tensor(t)
  assert_eq(out.get([0, 0]), 1.0)
  assert_eq(out.get([0, 1]), 2.0)
  let mlp = MLP::new(2, [2], 2, output_activation=act_softmax(), seed=2)
  let probs = mlp.mlp_predict_proba_tensor(t)
  let shape = probs.get_shape()
  assert_eq(shape[0], 1)
  assert_eq(shape[1], 2)
}
