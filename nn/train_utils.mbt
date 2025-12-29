///|
/// 训练工具：early stopping 与梯度裁剪

///|
pub enum EarlyStoppingMode {
  Min
  Max
}

///|
pub struct EarlyStopping {
  mode : EarlyStoppingMode
  patience : Int
  min_delta : Double
  mut best : Double
  mut num_bad : Int
  mut stopped : Bool
}

///|
pub fn EarlyStopping::new(
  mode : EarlyStoppingMode,
  patience : Int,
  min_delta? : Double = 0.0,
) -> EarlyStopping {
  if patience <= 0 {
    abort("patience 必须大于 0")
  }
  if min_delta < 0.0 {
    abort("min_delta 不能为负")
  }
  let best = match mode {
    Min => @double.infinity
    Max => -@double.infinity
  }
  { mode, patience, min_delta, best, num_bad: 0, stopped: false }
}

///|
pub fn early_stopping_min(
  patience : Int,
  min_delta? : Double = 0.0,
) -> EarlyStopping {
  EarlyStopping::new(Min, patience, min_delta~)
}

///|
pub fn early_stopping_max(
  patience : Int,
  min_delta? : Double = 0.0,
) -> EarlyStopping {
  EarlyStopping::new(Max, patience, min_delta~)
}

///|
pub fn EarlyStopping::reset(self : EarlyStopping) -> Unit {
  self.best = match self.mode {
    Min => @double.infinity
    Max => -@double.infinity
  }
  self.num_bad = 0
  self.stopped = false
}

///|
pub fn EarlyStopping::get_best(self : EarlyStopping) -> Double {
  self.best
}

///|
pub fn EarlyStopping::should_stop(self : EarlyStopping) -> Bool {
  self.stopped
}

///|
pub fn EarlyStopping::update(self : EarlyStopping, metric : Double) -> Bool {
  let improved = match self.mode {
    Min => metric < self.best - self.min_delta
    Max => metric > self.best + self.min_delta
  }
  if improved {
    self.best = metric
    self.num_bad = 0
    self.stopped = false
    return false
  }
  self.num_bad = self.num_bad + 1
  if self.num_bad >= self.patience {
    self.stopped = true
  }
  self.stopped
}

///|
/// 梯度裁剪：按范数缩放
pub fn clip_grad_norm(
  params : Array[OptimParam],
  max_norm : Double,
  eps? : Double = 1.0e-6,
) -> Double {
  if max_norm <= 0.0 {
    abort("max_norm 必须大于 0")
  }
  let mut sum = 0.0
  for i = 0; i < params.length(); i = i + 1 {
    let grads = params[i].grad
    for j = 0; j < grads.length(); j = j + 1 {
      let g = grads[j]
      sum = sum + g * g
    }
  }
  let total_norm = sum.sqrt()
  if total_norm > max_norm {
    let scale = max_norm / (total_norm + eps)
    for i = 0; i < params.length(); i = i + 1 {
      let grads = params[i].grad
      for j = 0; j < grads.length(); j = j + 1 {
        grads[j] = grads[j] * scale
      }
    }
  }
  total_norm
}

///|
/// 梯度裁剪：按绝对值截断
pub fn clip_grad_value(params : Array[OptimParam], clip_value : Double) -> Unit {
  if clip_value <= 0.0 {
    abort("clip_value 必须大于 0")
  }
  for i = 0; i < params.length(); i = i + 1 {
    let grads = params[i].grad
    for j = 0; j < grads.length(); j = j + 1 {
      let g = grads[j]
      if g > clip_value {
        grads[j] = clip_value
      } else if g < -clip_value {
        grads[j] = -clip_value
      }
    }
  }
}
