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

///|
/// 标准化训练一步：forward + loss + backward + step
fn batch_labels_to_indices(labels : Array[Double]) -> Array[Int] {
  let out = Array::make(labels.length(), 0)
  for i = 0; i < labels.length(); i = i + 1 {
    let v = labels[i]
    let idx = v.to_int()
    if idx < 0 || idx.to_double() != v {
      abort("labels 需为非负整数类标")
    }
    out[i] = idx
  }
  out
}

///|
fn batch_features_to_autotensor(features : Array[Array[Double]]) -> AutoTensor {
  AutoTensor::new(@math.Tensor::from_matrix(@math.Matrix::new(features)))
}

///|
pub fn train_step_batch(
  model : Sequential,
  batch : @data.Batch,
  opt : OptimizerParams,
) -> (Double, OptimizerParams) {
  match batch.labels {
    @data.Single(labels) => {
      if batch.features.length() != labels.length() {
        abort("batch features 和 labels 的样本数不匹配")
      }
      let x = batch_features_to_autotensor(batch.features)
      let y_true = batch_labels_to_indices(labels)
      train_step(model, x, y_true, opt)
    }
    @data.Multi(_) => abort("train_step_batch 仅支持单标签分类数据")
  }
}

///|
pub fn train_step(
  model : Sequential,
  x : AutoTensor,
  y_true : Array[Int],
  opt : OptimizerParams,
) -> (Double, OptimizerParams) {
  model.train()
  let opt2 = optimizer_params_zero_grad(opt)
  let (logits, ctx) = model.forward(x)
  let loss = logits.softmax_cross_entropy(y_true)
  loss.backward()
  model.sync_grads(ctx)
  let opt3 = optimizer_params_step(opt2)
  (loss.value().get([0]), opt3)
}

///|
/// 标准化评估一步：forward + loss
pub fn eval_step_batch(model : Sequential, batch : @data.Batch) -> Double {
  match batch.labels {
    @data.Single(labels) => {
      if batch.features.length() != labels.length() {
        abort("batch features 和 labels 的样本数不匹配")
      }
      let x = batch_features_to_autotensor(batch.features)
      let y_true = batch_labels_to_indices(labels)
      eval_step(model, x, y_true)
    }
    @data.Multi(_) => abort("eval_step_batch 仅支持单标签分类数据")
  }
}

///|
pub fn eval_step(
  model : Sequential,
  x : AutoTensor,
  y_true : Array[Int],
) -> Double {
  model.eval()
  let (logits, _ctx) = model.forward(x)
  let loss = logits.softmax_cross_entropy(y_true)
  loss.value().get([0])
}
