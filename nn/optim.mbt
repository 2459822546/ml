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
