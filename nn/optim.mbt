///|
/// 简单优化器（无自动求导，手写参数更新）

///|
pub struct OptimParam {
  data : Array[Double]
  grad : Array[Double]
}

///|
pub fn OptimParam::new(data : Array[Double]) -> OptimParam {
  { data, grad: Array::make(data.length(), 0.0) }
}

///|
pub fn OptimParam::zero_grad(self : OptimParam) -> Unit {
  for i = 0; i < self.grad.length(); i = i + 1 {
    self.grad[i] = 0.0
  }
}

///|
pub struct SGD {
  mut lr : Double
  momentum : Double
  weight_decay : Double
  velocity : Array[Double]
}

///|
pub fn SGD::new(
  lr : Double,
  size : Int,
  momentum? : Double = 0.0,
  weight_decay? : Double = 0.0,
) -> SGD {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  if momentum < 0.0 {
    abort("momentum 不能为负")
  }
  if weight_decay < 0.0 {
    abort("weight_decay 不能为负")
  }
  { lr, momentum, weight_decay, velocity: Array::make(size, 0.0) }
}

///|
pub fn SGD::step(self : SGD, param : OptimParam) -> Unit {
  if param.data.length() != param.grad.length() {
    abort("data/grad 长度不一致")
  }
  if self.velocity.length() != param.data.length() {
    abort("velocity 尺寸不匹配")
  }
  for i = 0; i < param.data.length(); i = i + 1 {
    let g = param.grad[i] + self.weight_decay * param.data[i]
    let v = self.momentum * self.velocity[i] - self.lr * g
    self.velocity[i] = v
    param.data[i] = param.data[i] + v
  }
}

///|
pub fn SGD::set_lr(self : SGD, lr : Double) -> Unit {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  self.lr = lr
}

///|
pub fn SGD::get_lr(self : SGD) -> Double {
  self.lr
}

///|
pub struct Adam {
  mut lr : Double
  beta1 : Double
  beta2 : Double
  eps : Double
  m : Array[Double]
  v : Array[Double]
  mut t : Int
}

///|
pub fn OptimParam::with_grad(
  self : OptimParam,
  values : Array[Double],
) -> OptimParam {
  if values.length() != self.grad.length() {
    abort("grad 长度不匹配")
  }
  { data: self.data, grad: values }
}

///|
pub fn Adam::new(
  lr : Double,
  size : Int,
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  eps? : Double = 1.0e-8,
) -> Adam {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  {
    lr,
    beta1,
    beta2,
    eps,
    m: Array::make(size, 0.0),
    v: Array::make(size, 0.0),
    t: 0,
  }
}

///|
pub fn Adam::step(self : Adam, param : OptimParam) -> Unit {
  if param.data.length() != param.grad.length() {
    abort("data/grad 长度不一致")
  }
  if self.m.length() != param.data.length() {
    abort("m 尺寸不匹配")
  }
  self.t = self.t + 1
  let t_hat = self.t.to_double()
  for i = 0; i < param.data.length(); i = i + 1 {
    let g = param.grad[i]
    self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
    self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g
    let m_hat = self.m[i] /
      (1.0 - @moonbitlang/core/math.pow(self.beta1, t_hat))
    let v_hat = self.v[i] /
      (1.0 - @moonbitlang/core/math.pow(self.beta2, t_hat))
    param.data[i] = param.data[i] -
      self.lr * m_hat / (@moonbitlang/core/math.pow(v_hat, 0.5) + self.eps)
  }
}

///|
pub fn Adam::set_lr(self : Adam, lr : Double) -> Unit {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  self.lr = lr
}

///|
pub fn Adam::get_lr(self : Adam) -> Double {
  self.lr
}

///|
pub struct AdamW {
  mut lr : Double
  beta1 : Double
  beta2 : Double
  eps : Double
  weight_decay : Double
  m : Array[Double]
  v : Array[Double]
  mut t : Int
}

///|
pub fn AdamW::new(
  lr : Double,
  size : Int,
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  eps? : Double = 1.0e-8,
  weight_decay? : Double = 0.0,
) -> AdamW {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  if weight_decay < 0.0 {
    abort("weight_decay 不能为负")
  }
  {
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    m: Array::make(size, 0.0),
    v: Array::make(size, 0.0),
    t: 0,
  }
}

///|
pub fn AdamW::step(self : AdamW, param : OptimParam) -> Unit {
  if param.data.length() != param.grad.length() {
    abort("data/grad 长度不一致")
  }
  if self.m.length() != param.data.length() {
    abort("m 尺寸不匹配")
  }
  self.t = self.t + 1
  let t_hat = self.t.to_double()
  for i = 0; i < param.data.length(); i = i + 1 {
    let g = param.grad[i]
    self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
    self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g
    let m_hat = self.m[i] /
      (1.0 - @moonbitlang/core/math.pow(self.beta1, t_hat))
    let v_hat = self.v[i] /
      (1.0 - @moonbitlang/core/math.pow(self.beta2, t_hat))
    let update = m_hat / (@moonbitlang/core/math.pow(v_hat, 0.5) + self.eps)
    param.data[i] = param.data[i] -
      self.lr * update -
      self.lr * self.weight_decay * param.data[i]
  }
}

///|
pub fn AdamW::set_lr(self : AdamW, lr : Double) -> Unit {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  self.lr = lr
}

///|
pub fn AdamW::get_lr(self : AdamW) -> Double {
  self.lr
}

///|
pub struct RMSProp {
  mut lr : Double
  alpha : Double
  eps : Double
  v : Array[Double]
}

///|
pub fn RMSProp::new(
  lr : Double,
  size : Int,
  alpha? : Double = 0.99,
  eps? : Double = 1.0e-8,
) -> RMSProp {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  if alpha <= 0.0 || alpha >= 1.0 {
    abort("alpha 必须在 (0, 1) 之间")
  }
  { lr, alpha, eps, v: Array::make(size, 0.0) }
}

///|
pub fn RMSProp::step(self : RMSProp, param : OptimParam) -> Unit {
  if param.data.length() != param.grad.length() {
    abort("data/grad 长度不一致")
  }
  if self.v.length() != param.data.length() {
    abort("v 尺寸不匹配")
  }
  for i = 0; i < param.data.length(); i = i + 1 {
    let g = param.grad[i]
    self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * g * g
    let denom = @moonbitlang/core/math.pow(self.v[i], 0.5) + self.eps
    param.data[i] = param.data[i] - self.lr * g / denom
  }
}

///|
pub fn RMSProp::set_lr(self : RMSProp, lr : Double) -> Unit {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  self.lr = lr
}

///|
pub fn RMSProp::get_lr(self : RMSProp) -> Double {
  self.lr
}

///|
pub struct Adagrad {
  mut lr : Double
  eps : Double
  h : Array[Double]
}

///|
pub fn Adagrad::new(lr : Double, size : Int, eps? : Double = 1.0e-8) -> Adagrad {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  { lr, eps, h: Array::make(size, 0.0) }
}

///|
pub fn Adagrad::step(self : Adagrad, param : OptimParam) -> Unit {
  if param.data.length() != param.grad.length() {
    abort("data/grad 长度不一致")
  }
  if self.h.length() != param.data.length() {
    abort("h 尺寸不匹配")
  }
  for i = 0; i < param.data.length(); i = i + 1 {
    let g = param.grad[i]
    self.h[i] = self.h[i] + g * g
    let denom = @moonbitlang/core/math.pow(self.h[i], 0.5) + self.eps
    param.data[i] = param.data[i] - self.lr * g / denom
  }
}

///|
pub fn Adagrad::set_lr(self : Adagrad, lr : Double) -> Unit {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  self.lr = lr
}

///|
pub fn Adagrad::get_lr(self : Adagrad) -> Double {
  self.lr
}

///|
/// 学习率衰减：简单阶梯函数
pub fn step_decay(
  lr : Double,
  factor : Double,
  step_size : Int,
  epoch : Int,
) -> Double {
  if factor <= 0.0 || factor >= 1.0 {
    return lr
  }
  if step_size <= 0 {
    return lr
  }
  let k = epoch / step_size
  lr * @moonbitlang/core/math.pow(factor, k.to_double())
}

///|
fn tensor_numel(t : @math.Tensor) -> Int {
  let shape = t.get_shape()
  let mut total = 1
  for i = 0; i < shape.length(); i = i + 1 {
    total = total * shape[i]
  }
  total
}

///|
fn optim_compute_strides(shape : Array[Int]) -> Array[Int] {
  let strides = Array::make(shape.length(), 0)
  let mut stride = 1
  for i = shape.length() - 1; i >= 0; i = i - 1 {
    strides[i] = stride
    stride = stride * shape[i]
  }
  strides
}

///|
fn optim_indices_from_flat(
  shape : Array[Int],
  strides : Array[Int],
  idx : Int,
) -> Array[Int] {
  let coords = Array::make(shape.length(), 0)
  let mut remain = idx
  for i = 0; i < shape.length(); i = i + 1 {
    coords[i] = remain / strides[i]
    remain = remain % strides[i]
  }
  coords
}

///|
pub struct SGDParams {
  mut lr : Double
  mut momentum : Double
  mut weight_decay : Double
  mut velocities : Array[Array[Double]]
}

///|
pub fn SGDParams::new(
  lr : Double,
  params : Array[Parameter],
  momentum? : Double = 0.0,
  weight_decay? : Double = 0.0,
) -> SGDParams {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  if momentum < 0.0 {
    abort("momentum 不能为负")
  }
  if weight_decay < 0.0 {
    abort("weight_decay 不能为负")
  }
  let velocities = Array::make(params.length(), [])
  for i = 0; i < params.length(); i = i + 1 {
    let total = tensor_numel(params[i].get_data())
    velocities[i] = Array::make(total, 0.0)
  }
  { lr, momentum, weight_decay, velocities }
}

///|
pub fn SGDParams::step(self : SGDParams, params : Array[Parameter]) -> Unit {
  if params.length() != self.velocities.length() {
    abort("参数数量不匹配")
  }
  for i = 0; i < params.length(); i = i + 1 {
    let data = params[i].get_data()
    let grad = params[i].get_grad()
    let shape = data.get_shape()
    if shape != grad.get_shape() {
      abort("data/grad 形状不一致")
    }
    let total = tensor_numel(data)
    if self.velocities[i].length() != total {
      abort("velocity 尺寸不匹配")
    }
    let strides = optim_compute_strides(shape)
    for idx = 0; idx < total; idx = idx + 1 {
      let coord = optim_indices_from_flat(shape, strides, idx)
      let g = grad.get(coord) + self.weight_decay * data.get(coord)
      let v = self.momentum * self.velocities[i][idx] - self.lr * g
      self.velocities[i][idx] = v
      data.set(coord, data.get(coord) + v)
    }
    params[i].set_data_inplace(data)
  }
}

///|
pub fn SGDParams::step_seq(self : SGDParams, seq : Sequential) -> Unit {
  self.step(seq.parameters())
}

///|
pub struct AdamParams {
  mut lr : Double
  mut beta1 : Double
  mut beta2 : Double
  mut eps : Double
  mut m : Array[Array[Double]]
  mut v : Array[Array[Double]]
  mut t : Int
}

///|
pub fn AdamParams::new(
  lr : Double,
  params : Array[Parameter],
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  eps? : Double = 1.0e-8,
) -> AdamParams {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  let m = Array::make(params.length(), [])
  let v = Array::make(params.length(), [])
  for i = 0; i < params.length(); i = i + 1 {
    let total = tensor_numel(params[i].get_data())
    m[i] = Array::make(total, 0.0)
    v[i] = Array::make(total, 0.0)
  }
  { lr, beta1, beta2, eps, m, v, t: 0 }
}

///|
pub fn AdamParams::step(self : AdamParams, params : Array[Parameter]) -> Unit {
  if params.length() != self.m.length() {
    abort("参数数量不匹配")
  }
  self.t = self.t + 1
  let t_hat = self.t.to_double()
  for i = 0; i < params.length(); i = i + 1 {
    let data = params[i].get_data()
    let grad = params[i].get_grad()
    let shape = data.get_shape()
    if shape != grad.get_shape() {
      abort("data/grad 形状不一致")
    }
    let total = tensor_numel(data)
    let strides = optim_compute_strides(shape)
    for idx = 0; idx < total; idx = idx + 1 {
      let coord = optim_indices_from_flat(shape, strides, idx)
      let g = grad.get(coord)
      self.m[i][idx] = self.beta1 * self.m[i][idx] + (1.0 - self.beta1) * g
      self.v[i][idx] = self.beta2 * self.v[i][idx] + (1.0 - self.beta2) * g * g
      let m_hat = self.m[i][idx] /
        (1.0 - @moonbitlang/core/math.pow(self.beta1, t_hat))
      let v_hat = self.v[i][idx] /
        (1.0 - @moonbitlang/core/math.pow(self.beta2, t_hat))
      data.set(
        coord,
        data.get(coord) -
        self.lr * m_hat / (@moonbitlang/core/math.pow(v_hat, 0.5) + self.eps),
      )
    }
    params[i].set_data_inplace(data)
  }
}

///|
pub fn AdamParams::step_seq(self : AdamParams, seq : Sequential) -> Unit {
  self.step(seq.parameters())
}

///|
pub struct AdamWParams {
  mut lr : Double
  mut beta1 : Double
  mut beta2 : Double
  mut eps : Double
  mut weight_decay : Double
  mut m : Array[Array[Double]]
  mut v : Array[Array[Double]]
  mut t : Int
}

///|
pub fn AdamWParams::new(
  lr : Double,
  params : Array[Parameter],
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  eps? : Double = 1.0e-8,
  weight_decay? : Double = 0.0,
) -> AdamWParams {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  if weight_decay < 0.0 {
    abort("weight_decay 不能为负")
  }
  let m = Array::make(params.length(), [])
  let v = Array::make(params.length(), [])
  for i = 0; i < params.length(); i = i + 1 {
    let total = tensor_numel(params[i].get_data())
    m[i] = Array::make(total, 0.0)
    v[i] = Array::make(total, 0.0)
  }
  { lr, beta1, beta2, eps, weight_decay, m, v, t: 0 }
}

///|
pub fn AdamWParams::step(self : AdamWParams, params : Array[Parameter]) -> Unit {
  if params.length() != self.m.length() {
    abort("参数数量不匹配")
  }
  self.t = self.t + 1
  let t_hat = self.t.to_double()
  for i = 0; i < params.length(); i = i + 1 {
    let data = params[i].get_data()
    let grad = params[i].get_grad()
    let shape = data.get_shape()
    if shape != grad.get_shape() {
      abort("data/grad 形状不一致")
    }
    let total = tensor_numel(data)
    let strides = optim_compute_strides(shape)
    for idx = 0; idx < total; idx = idx + 1 {
      let coord = optim_indices_from_flat(shape, strides, idx)
      let g = grad.get(coord)
      self.m[i][idx] = self.beta1 * self.m[i][idx] + (1.0 - self.beta1) * g
      self.v[i][idx] = self.beta2 * self.v[i][idx] + (1.0 - self.beta2) * g * g
      let m_hat = self.m[i][idx] /
        (1.0 - @moonbitlang/core/math.pow(self.beta1, t_hat))
      let v_hat = self.v[i][idx] /
        (1.0 - @moonbitlang/core/math.pow(self.beta2, t_hat))
      let update = m_hat / (@moonbitlang/core/math.pow(v_hat, 0.5) + self.eps)
      data.set(
        coord,
        data.get(coord) -
        self.lr * update -
        self.lr * self.weight_decay * data.get(coord),
      )
    }
    params[i].set_data_inplace(data)
  }
}

///|
pub fn AdamWParams::step_seq(self : AdamWParams, seq : Sequential) -> Unit {
  self.step(seq.parameters())
}

///|
pub struct RMSPropParams {
  mut lr : Double
  mut alpha : Double
  mut eps : Double
  mut v : Array[Array[Double]]
}

///|
pub fn RMSPropParams::new(
  lr : Double,
  params : Array[Parameter],
  alpha? : Double = 0.99,
  eps? : Double = 1.0e-8,
) -> RMSPropParams {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  if alpha <= 0.0 || alpha >= 1.0 {
    abort("alpha 必须在 (0, 1) 之间")
  }
  let v = Array::make(params.length(), [])
  for i = 0; i < params.length(); i = i + 1 {
    let total = tensor_numel(params[i].get_data())
    v[i] = Array::make(total, 0.0)
  }
  { lr, alpha, eps, v }
}

///|
pub fn RMSPropParams::step(
  self : RMSPropParams,
  params : Array[Parameter],
) -> Unit {
  if params.length() != self.v.length() {
    abort("参数数量不匹配")
  }
  for i = 0; i < params.length(); i = i + 1 {
    let data = params[i].get_data()
    let grad = params[i].get_grad()
    let shape = data.get_shape()
    if shape != grad.get_shape() {
      abort("data/grad 形状不一致")
    }
    let total = tensor_numel(data)
    let strides = optim_compute_strides(shape)
    for idx = 0; idx < total; idx = idx + 1 {
      let coord = optim_indices_from_flat(shape, strides, idx)
      let g = grad.get(coord)
      self.v[i][idx] = self.alpha * self.v[i][idx] + (1.0 - self.alpha) * g * g
      let denom = @moonbitlang/core/math.pow(self.v[i][idx], 0.5) + self.eps
      data.set(coord, data.get(coord) - self.lr * g / denom)
    }
    params[i].set_data_inplace(data)
  }
}

///|
pub fn RMSPropParams::step_seq(self : RMSPropParams, seq : Sequential) -> Unit {
  self.step(seq.parameters())
}

///|
pub struct AdagradParams {
  mut lr : Double
  mut eps : Double
  mut h : Array[Array[Double]]
}

///|
pub fn AdagradParams::new(
  lr : Double,
  params : Array[Parameter],
  eps? : Double = 1.0e-8,
) -> AdagradParams {
  if lr <= 0.0 {
    abort("lr 必须大于 0")
  }
  let h = Array::make(params.length(), [])
  for i = 0; i < params.length(); i = i + 1 {
    let total = tensor_numel(params[i].get_data())
    h[i] = Array::make(total, 0.0)
  }
  { lr, eps, h }
}

///|
pub fn AdagradParams::step(
  self : AdagradParams,
  params : Array[Parameter],
) -> Unit {
  if params.length() != self.h.length() {
    abort("参数数量不匹配")
  }
  for i = 0; i < params.length(); i = i + 1 {
    let data = params[i].get_data()
    let grad = params[i].get_grad()
    let shape = data.get_shape()
    if shape != grad.get_shape() {
      abort("data/grad 形状不一致")
    }
    let total = tensor_numel(data)
    let strides = optim_compute_strides(shape)
    for idx = 0; idx < total; idx = idx + 1 {
      let coord = optim_indices_from_flat(shape, strides, idx)
      let g = grad.get(coord)
      self.h[i][idx] = self.h[i][idx] + g * g
      let denom = @moonbitlang/core/math.pow(self.h[i][idx], 0.5) + self.eps
      data.set(coord, data.get(coord) - self.lr * g / denom)
    }
    params[i].set_data_inplace(data)
  }
}

///|
pub fn AdagradParams::step_seq(self : AdagradParams, seq : Sequential) -> Unit {
  self.step(seq.parameters())
}

///|
/// 余弦学习率衰减
pub fn cosine_decay(
  lr : Double,
  min_lr : Double,
  epoch : Int,
  max_epoch : Int,
) -> Double {
  if max_epoch <= 0 {
    return lr
  }
  let e = if epoch < 0 {
    0
  } else if epoch > max_epoch {
    max_epoch
  } else {
    epoch
  }
  let t = e.to_double()
  let t_max = max_epoch.to_double()
  let pi = 3.141592653589793
  let cosine = @moonbitlang/core/math.cos(pi * t / t_max)
  min_lr + 0.5 * (lr - min_lr) * (1.0 + cosine)
}

///|
/// 线性 warmup
pub fn linear_warmup(lr : Double, warmup_steps : Int, step : Int) -> Double {
  if warmup_steps <= 0 {
    return lr
  }
  if step <= 0 {
    return 0.0
  }
  if step >= warmup_steps {
    return lr
  }
  lr * (step.to_double() / warmup_steps.to_double())
}

///|
/// warmup + 余弦衰减
pub fn warmup_cosine(
  lr : Double,
  min_lr : Double,
  warmup_steps : Int,
  step : Int,
  total_steps : Int,
) -> Double {
  if total_steps <= 0 {
    return lr
  }
  if step < warmup_steps {
    return linear_warmup(lr, warmup_steps, step)
  }
  let decay_steps = total_steps - warmup_steps
  if decay_steps <= 0 {
    return lr
  }
  let t = (step - warmup_steps).to_double()
  let t_max = decay_steps.to_double()
  let progress = if t > t_max { t_max } else { t }
  let pi = 3.141592653589793
  let cosine = @moonbitlang/core/math.cos(pi * progress / t_max)
  min_lr + 0.5 * (lr - min_lr) * (1.0 + cosine)
}

///|
/// 统一 Optimizer 接口
pub enum Optimizer {
  SGD(SGD)
  Adam(Adam)
  AdamW(AdamW)
  RMSProp(RMSProp)
  Adagrad(Adagrad)
}

///|
pub fn optimizer_sgd(opt : SGD) -> Optimizer {
  SGD(opt)
}

///|
pub fn optimizer_adam(opt : Adam) -> Optimizer {
  Adam(opt)
}

///|
pub fn optimizer_adamw(opt : AdamW) -> Optimizer {
  AdamW(opt)
}

///|
pub fn optimizer_rmsprop(opt : RMSProp) -> Optimizer {
  RMSProp(opt)
}

///|
pub fn optimizer_adagrad(opt : Adagrad) -> Optimizer {
  Adagrad(opt)
}

///|
pub fn optimizer_step(opt : Optimizer, param : OptimParam) -> Optimizer {
  match opt {
    SGD(s) => {
      let o = s
      o.step(param)
      SGD(o)
    }
    Adam(a) => {
      let o = a
      o.step(param)
      Adam(o)
    }
    AdamW(a) => {
      let o = a
      o.step(param)
      AdamW(o)
    }
    RMSProp(r) => {
      let o = r
      o.step(param)
      RMSProp(o)
    }
    Adagrad(a) => {
      let o = a
      o.step(param)
      Adagrad(o)
    }
  }
}

///|
pub fn optimizer_set_lr(opt : Optimizer, lr : Double) -> Optimizer {
  match opt {
    SGD(s) => {
      let o = s
      o.set_lr(lr)
      SGD(o)
    }
    Adam(a) => {
      let o = a
      o.set_lr(lr)
      Adam(o)
    }
    AdamW(a) => {
      let o = a
      o.set_lr(lr)
      AdamW(o)
    }
    RMSProp(r) => {
      let o = r
      o.set_lr(lr)
      RMSProp(o)
    }
    Adagrad(a) => {
      let o = a
      o.set_lr(lr)
      Adagrad(o)
    }
  }
}

///|
pub fn optimizer_get_lr(opt : Optimizer) -> Double {
  match opt {
    SGD(s) => s.get_lr()
    Adam(a) => a.get_lr()
    AdamW(a) => a.get_lr()
    RMSProp(r) => r.get_lr()
    Adagrad(a) => a.get_lr()
  }
}

///|
/// 统一 Parameter 优化器接口
pub enum OptimizerParams {
  SGD(SGDParams, Array[Parameter])
  Adam(AdamParams, Array[Parameter])
  AdamW(AdamWParams, Array[Parameter])
  RMSProp(RMSPropParams, Array[Parameter])
  Adagrad(AdagradParams, Array[Parameter])
}

///|
pub fn optimizer_params_sgd(
  params : Array[Parameter],
  lr : Double,
  momentum? : Double = 0.0,
  weight_decay? : Double = 0.0,
) -> OptimizerParams {
  SGD(SGDParams::new(lr, params, momentum~, weight_decay~), params)
}

///|
pub fn optimizer_params_adam(
  params : Array[Parameter],
  lr : Double,
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  eps? : Double = 1.0e-8,
) -> OptimizerParams {
  Adam(AdamParams::new(lr, params, beta1~, beta2~, eps~), params)
}

///|
pub fn optimizer_params_adamw(
  params : Array[Parameter],
  lr : Double,
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  eps? : Double = 1.0e-8,
  weight_decay? : Double = 0.0,
) -> OptimizerParams {
  AdamW(
    AdamWParams::new(lr, params, beta1~, beta2~, eps~, weight_decay~),
    params,
  )
}

///|
pub fn optimizer_params_rmsprop(
  params : Array[Parameter],
  lr : Double,
  alpha? : Double = 0.99,
  eps? : Double = 1.0e-8,
) -> OptimizerParams {
  RMSProp(RMSPropParams::new(lr, params, alpha~, eps~), params)
}

///|
pub fn optimizer_params_adagrad(
  params : Array[Parameter],
  lr : Double,
  eps? : Double = 1.0e-8,
) -> OptimizerParams {
  Adagrad(AdagradParams::new(lr, params, eps~), params)
}

///|
pub fn optimizer_params_zero_grad(opt : OptimizerParams) -> OptimizerParams {
  match opt {
    SGD(s, params) => {
      for i = 0; i < params.length(); i = i + 1 {
        params[i].zero_grad()
      }
      SGD(s, params)
    }
    Adam(a, params) => {
      for i = 0; i < params.length(); i = i + 1 {
        params[i].zero_grad()
      }
      Adam(a, params)
    }
    AdamW(a, params) => {
      for i = 0; i < params.length(); i = i + 1 {
        params[i].zero_grad()
      }
      AdamW(a, params)
    }
    RMSProp(r, params) => {
      for i = 0; i < params.length(); i = i + 1 {
        params[i].zero_grad()
      }
      RMSProp(r, params)
    }
    Adagrad(a, params) => {
      for i = 0; i < params.length(); i = i + 1 {
        params[i].zero_grad()
      }
      Adagrad(a, params)
    }
  }
}

///|
pub fn optimizer_params_step(opt : OptimizerParams) -> OptimizerParams {
  match opt {
    SGD(s, params) => {
      let o = s
      o.step(params)
      SGD(o, params)
    }
    Adam(a, params) => {
      let o = a
      o.step(params)
      Adam(o, params)
    }
    AdamW(a, params) => {
      let o = a
      o.step(params)
      AdamW(o, params)
    }
    RMSProp(r, params) => {
      let o = r
      o.step(params)
      RMSProp(o, params)
    }
    Adagrad(a, params) => {
      let o = a
      o.step(params)
      Adagrad(o, params)
    }
  }
}

///|
pub fn OptimizerParams::zero_grad(self : OptimizerParams) -> OptimizerParams {
  optimizer_params_zero_grad(self)
}

///|
pub fn OptimizerParams::step(self : OptimizerParams) -> OptimizerParams {
  optimizer_params_step(self)
}

///|
pub fn optimizer_params_state_dict(
  opt : OptimizerParams,
) -> Map[String, @math.Tensor] {
  let dict = Map::new()
  match opt {
    SGD(s, _) => {
      dict.set("type", @math.Tensor::new([0.0], [1]))
      dict.set("lr", @math.Tensor::new([s.lr], [1]))
      dict.set("momentum", @math.Tensor::new([s.momentum], [1]))
      dict.set("weight_decay", @math.Tensor::new([s.weight_decay], [1]))
      dict.set(
        "velocity",
        @math.Tensor::new(tensor_concat_flat(s.velocities), [
          tensor_total_len(s.velocities),
        ]),
      )
      dict.set("velocity_shapes", tensor_pack_shapes(s.velocities))
    }
    Adam(a, _) => {
      dict.set("type", @math.Tensor::new([1.0], [1]))
      dict.set("lr", @math.Tensor::new([a.lr], [1]))
      dict.set("beta1", @math.Tensor::new([a.beta1], [1]))
      dict.set("beta2", @math.Tensor::new([a.beta2], [1]))
      dict.set("eps", @math.Tensor::new([a.eps], [1]))
      dict.set("t", @math.Tensor::new([a.t.to_double()], [1]))
      dict.set(
        "m",
        @math.Tensor::new(tensor_concat_flat(a.m), [tensor_total_len(a.m)]),
      )
      dict.set(
        "v",
        @math.Tensor::new(tensor_concat_flat(a.v), [tensor_total_len(a.v)]),
      )
      dict.set("m_shapes", tensor_pack_shapes(a.m))
      dict.set("v_shapes", tensor_pack_shapes(a.v))
    }
    AdamW(a, _) => {
      dict.set("type", @math.Tensor::new([2.0], [1]))
      dict.set("lr", @math.Tensor::new([a.lr], [1]))
      dict.set("beta1", @math.Tensor::new([a.beta1], [1]))
      dict.set("beta2", @math.Tensor::new([a.beta2], [1]))
      dict.set("eps", @math.Tensor::new([a.eps], [1]))
      dict.set("weight_decay", @math.Tensor::new([a.weight_decay], [1]))
      dict.set("t", @math.Tensor::new([a.t.to_double()], [1]))
      dict.set(
        "m",
        @math.Tensor::new(tensor_concat_flat(a.m), [tensor_total_len(a.m)]),
      )
      dict.set(
        "v",
        @math.Tensor::new(tensor_concat_flat(a.v), [tensor_total_len(a.v)]),
      )
      dict.set("m_shapes", tensor_pack_shapes(a.m))
      dict.set("v_shapes", tensor_pack_shapes(a.v))
    }
    RMSProp(r, _) => {
      dict.set("type", @math.Tensor::new([3.0], [1]))
      dict.set("lr", @math.Tensor::new([r.lr], [1]))
      dict.set("alpha", @math.Tensor::new([r.alpha], [1]))
      dict.set("eps", @math.Tensor::new([r.eps], [1]))
      dict.set(
        "v",
        @math.Tensor::new(tensor_concat_flat(r.v), [tensor_total_len(r.v)]),
      )
      dict.set("v_shapes", tensor_pack_shapes(r.v))
    }
    Adagrad(a, _) => {
      dict.set("type", @math.Tensor::new([4.0], [1]))
      dict.set("lr", @math.Tensor::new([a.lr], [1]))
      dict.set("eps", @math.Tensor::new([a.eps], [1]))
      dict.set(
        "h",
        @math.Tensor::new(tensor_concat_flat(a.h), [tensor_total_len(a.h)]),
      )
      dict.set("h_shapes", tensor_pack_shapes(a.h))
    }
  }
  dict
}

///|
pub fn optimizer_params_load_state_dict(
  opt : OptimizerParams,
  dict : Map[String, @math.Tensor],
  strict? : Bool = true,
) -> OptimizerParams {
  let strict_val = strict
  fn get_req(
    dict : Map[String, @math.Tensor],
    key : String,
    strict : Bool,
  ) -> @math.Tensor? {
    match dict.get(key) {
      Some(v) => Some(v)
      None =>
        if strict {
          abort("optimizer state 缺少键: " + key)
        } else {
          None
        }
    }
  }

  let load_type = match dict.get("type") {
    Some(v) => v.get([0]).to_int()
    None =>
      if strict_val {
        abort("optimizer state 缺少键: type")
      } else {
        -1
      }
  }
  match opt {
    SGD(s, params) => {
      if strict_val && load_type != 0 {
        abort("optimizer type 不匹配")
      }
      let lr = get_req(dict, "lr", strict_val)
      let momentum = get_req(dict, "momentum", strict_val)
      let weight_decay = get_req(dict, "weight_decay", strict_val)
      let velocity = get_req(dict, "velocity", strict_val)
      let shapes = get_req(dict, "velocity_shapes", strict_val)
      match (lr, momentum, weight_decay, velocity, shapes) {
        (Some(lr_t), Some(mom_t), Some(wd_t), Some(v_t), Some(s_t)) => {
          s.lr = lr_t.get([0])
          s.momentum = mom_t.get([0])
          s.weight_decay = wd_t.get([0])
          s.velocities = tensor_unpack_shapes(v_t, s_t)
        }
        _ => ()
      }
      SGD(s, params)
    }
    Adam(a, params) => {
      if strict_val && load_type != 1 {
        abort("optimizer type 不匹配")
      }
      let lr = get_req(dict, "lr", strict_val)
      let beta1 = get_req(dict, "beta1", strict_val)
      let beta2 = get_req(dict, "beta2", strict_val)
      let eps = get_req(dict, "eps", strict_val)
      let t = get_req(dict, "t", strict_val)
      let m = get_req(dict, "m", strict_val)
      let v = get_req(dict, "v", strict_val)
      let m_shapes = get_req(dict, "m_shapes", strict_val)
      let v_shapes = get_req(dict, "v_shapes", strict_val)
      match (lr, beta1, beta2, eps, t, m, v, m_shapes, v_shapes) {
        (
          Some(lr_t),
          Some(b1_t),
          Some(b2_t),
          Some(eps_t),
          Some(t_t),
          Some(m_t),
          Some(v_t),
          Some(ms_t),
          Some(vs_t),
        ) => {
          a.lr = lr_t.get([0])
          a.beta1 = b1_t.get([0])
          a.beta2 = b2_t.get([0])
          a.eps = eps_t.get([0])
          a.t = t_t.get([0]).to_int()
          a.m = tensor_unpack_shapes(m_t, ms_t)
          a.v = tensor_unpack_shapes(v_t, vs_t)
        }
        _ => ()
      }
      Adam(a, params)
    }
    AdamW(a, params) => {
      if strict_val && load_type != 2 {
        abort("optimizer type 不匹配")
      }
      let lr = get_req(dict, "lr", strict_val)
      let beta1 = get_req(dict, "beta1", strict_val)
      let beta2 = get_req(dict, "beta2", strict_val)
      let eps = get_req(dict, "eps", strict_val)
      let weight_decay = get_req(dict, "weight_decay", strict_val)
      let t = get_req(dict, "t", strict_val)
      let m = get_req(dict, "m", strict_val)
      let v = get_req(dict, "v", strict_val)
      let m_shapes = get_req(dict, "m_shapes", strict_val)
      let v_shapes = get_req(dict, "v_shapes", strict_val)
      match (lr, beta1, beta2, eps, weight_decay, t, m, v, m_shapes, v_shapes) {
        (
          Some(lr_t),
          Some(b1_t),
          Some(b2_t),
          Some(eps_t),
          Some(wd_t),
          Some(t_t),
          Some(m_t),
          Some(v_t),
          Some(ms_t),
          Some(vs_t),
        ) => {
          a.lr = lr_t.get([0])
          a.beta1 = b1_t.get([0])
          a.beta2 = b2_t.get([0])
          a.eps = eps_t.get([0])
          a.weight_decay = wd_t.get([0])
          a.t = t_t.get([0]).to_int()
          a.m = tensor_unpack_shapes(m_t, ms_t)
          a.v = tensor_unpack_shapes(v_t, vs_t)
        }
        _ => ()
      }
      AdamW(a, params)
    }
    RMSProp(r, params) => {
      if strict_val && load_type != 3 {
        abort("optimizer type 不匹配")
      }
      let lr = get_req(dict, "lr", strict_val)
      let alpha = get_req(dict, "alpha", strict_val)
      let eps = get_req(dict, "eps", strict_val)
      let v = get_req(dict, "v", strict_val)
      let v_shapes = get_req(dict, "v_shapes", strict_val)
      match (lr, alpha, eps, v, v_shapes) {
        (Some(lr_t), Some(a_t), Some(eps_t), Some(v_t), Some(vs_t)) => {
          r.lr = lr_t.get([0])
          r.alpha = a_t.get([0])
          r.eps = eps_t.get([0])
          r.v = tensor_unpack_shapes(v_t, vs_t)
        }
        _ => ()
      }
      RMSProp(r, params)
    }
    Adagrad(a, params) => {
      if strict_val && load_type != 4 {
        abort("optimizer type 不匹配")
      }
      let lr = get_req(dict, "lr", strict_val)
      let eps = get_req(dict, "eps", strict_val)
      let h = get_req(dict, "h", strict_val)
      let h_shapes = get_req(dict, "h_shapes", strict_val)
      match (lr, eps, h, h_shapes) {
        (Some(lr_t), Some(eps_t), Some(h_t), Some(hs_t)) => {
          a.lr = lr_t.get([0])
          a.eps = eps_t.get([0])
          a.h = tensor_unpack_shapes(h_t, hs_t)
        }
        _ => ()
      }
      Adagrad(a, params)
    }
  }
}

///|
fn tensor_total_len(xs : Array[Array[Double]]) -> Int {
  let mut total = 0
  for i = 0; i < xs.length(); i = i + 1 {
    total = total + xs[i].length()
  }
  total
}

///|
fn tensor_concat_flat(xs : Array[Array[Double]]) -> Array[Double] {
  let total = tensor_total_len(xs)
  let out = Array::make(total, 0.0)
  let mut idx = 0
  for i = 0; i < xs.length(); i = i + 1 {
    for j = 0; j < xs[i].length(); j = j + 1 {
      out[idx] = xs[i][j]
      idx = idx + 1
    }
  }
  out
}

///|
fn tensor_pack_shapes(xs : Array[Array[Double]]) -> @math.Tensor {
  let data = Array::make(xs.length(), 0.0)
  for i = 0; i < xs.length(); i = i + 1 {
    data[i] = xs[i].length().to_double()
  }
  @math.Tensor::new(data, [xs.length()])
}

///|
fn tensor_unpack_shapes(
  flat : @math.Tensor,
  shape_tensor : @math.Tensor,
) -> Array[Array[Double]] {
  let n = shape_tensor.get_shape()[0]
  let out = Array::make(n, [])
  let mut offset = 0
  for i = 0; i < n; i = i + 1 {
    let len = shape_tensor.get([i]).to_int()
    let arr = Array::make(len, 0.0)
    for j = 0; j < len; j = j + 1 {
      arr[j] = flat.get([offset + j])
    }
    out[i] = arr
    offset = offset + len
  }
  out
}
