///|
/// 训练工具：early stopping 与梯度裁剪

///|
pub struct TrainMetrics {
  mut loss_sum : Double
  mut loss_count : Int
  mut last_loss : Double
  mut sums : Map[String, Double]
  mut counts : Map[String, Int]
}

///|
pub fn TrainMetrics::new() -> TrainMetrics {
  {
    loss_sum: 0.0,
    loss_count: 0,
    last_loss: 0.0,
    sums: Map::new(),
    counts: Map::new(),
  }
}

///|
pub fn TrainMetrics::reset(self : TrainMetrics) -> Unit {
  self.loss_sum = 0.0
  self.loss_count = 0
  self.last_loss = 0.0
  self.sums = Map::new()
  self.counts = Map::new()
}

///|
pub fn TrainMetrics::update_loss(
  self : TrainMetrics,
  loss : Double,
  count : Int,
) -> Unit {
  if count <= 0 {
    abort("count 必须大于 0")
  }
  self.loss_sum = self.loss_sum + loss * count.to_double()
  self.loss_count = self.loss_count + count
  self.last_loss = loss
}

///|
pub fn TrainMetrics::loss_avg(self : TrainMetrics) -> Double {
  if self.loss_count == 0 {
    0.0
  } else {
    self.loss_sum / self.loss_count.to_double()
  }
}

///|
pub fn TrainMetrics::last(self : TrainMetrics) -> Double {
  self.last_loss
}

///|
pub fn TrainMetrics::add_metric_avg(
  self : TrainMetrics,
  name : String,
  value : Double,
  count? : Int = 1,
) -> Unit {
  if count <= 0 {
    abort("count 必须大于 0")
  }
  let sum = match self.sums.get(name) {
    Some(v) => v
    None => 0.0
  }
  let cnt = match self.counts.get(name) {
    Some(v) => v
    None => 0
  }
  self.sums.set(name, sum + value * count.to_double())
  self.counts.set(name, cnt + count)
}

///|
pub fn TrainMetrics::metric_avg(self : TrainMetrics, name : String) -> Double {
  match self.counts.get(name) {
    Some(cnt) =>
      if cnt == 0 {
        0.0
      } else {
        match self.sums.get(name) {
          Some(sum) => sum / cnt.to_double()
          None => 0.0
        }
      }
    None => 0.0
  }
}

///|
pub fn TrainMetrics::add_metrics_avg(
  self : TrainMetrics,
  metrics : Map[String, Double],
  count : Int,
) -> Unit {
  for key in metrics.keys() {
    match metrics.get(key) {
      Some(value) => self.add_metric_avg(key, value, count~)
      None => ()
    }
  }
}

///|
pub struct TrainCallbacks {
  on_batch_end : (Int, Int, Double, TrainMetrics, OptimizerParams) -> Unit
  on_epoch_end : (Int, TrainMetrics, OptimizerParams) -> Unit
}

///|
pub fn TrainCallbacks::none() -> TrainCallbacks {
  {
    on_batch_end: fn(
      _epoch : Int,
      _batch : Int,
      _loss : Double,
      _m : TrainMetrics,
      _opt : OptimizerParams,
    ) -> Unit {
      ()
    },
    on_epoch_end: fn(
      _epoch : Int,
      _m : TrainMetrics,
      _opt : OptimizerParams,
    ) -> Unit {
      ()
    },
  }
}

///|
pub enum LRScheduler {
  None
  StepLR(StepLRScheduler)
  CosineLR(CosineLRScheduler)
  WarmupCosineLR(WarmupCosineLRScheduler)
}

///|
pub struct StepLRScheduler {
  base_lr : Double
  step_size : Int
  gamma : Double
  mut step : Int
}

///|
pub struct CosineLRScheduler {
  base_lr : Double
  min_lr : Double
  total_steps : Int
  mut step : Int
}

///|
pub struct WarmupCosineLRScheduler {
  base_lr : Double
  min_lr : Double
  total_steps : Int
  warmup_steps : Int
  mut step : Int
}

///|
pub fn step_lr(
  base_lr : Double,
  step_size : Int,
  gamma : Double,
) -> LRScheduler {
  if base_lr <= 0.0 {
    abort("base_lr 必须大于 0")
  }
  if step_size <= 0 {
    abort("step_size 必须大于 0")
  }
  if gamma <= 0.0 {
    abort("gamma 必须大于 0")
  }
  StepLR({ base_lr, step_size, gamma, step: 0 })
}

///|
pub fn cosine_lr(
  base_lr : Double,
  total_steps : Int,
  min_lr? : Double = 0.0,
) -> LRScheduler {
  if base_lr <= 0.0 {
    abort("base_lr 必须大于 0")
  }
  if total_steps <= 0 {
    abort("total_steps 必须大于 0")
  }
  if min_lr < 0.0 {
    abort("min_lr 不能为负")
  }
  CosineLR({ base_lr, min_lr, total_steps, step: 0 })
}

///|
pub fn warmup_cosine_lr(
  base_lr : Double,
  total_steps : Int,
  warmup_steps : Int,
  min_lr? : Double = 0.0,
) -> LRScheduler {
  if base_lr <= 0.0 {
    abort("base_lr 必须大于 0")
  }
  if total_steps <= 0 {
    abort("total_steps 必须大于 0")
  }
  if warmup_steps < 0 {
    abort("warmup_steps 不能为负")
  }
  if min_lr < 0.0 {
    abort("min_lr 不能为负")
  }
  WarmupCosineLR({ base_lr, min_lr, total_steps, warmup_steps, step: 0 })
}

///|
pub fn lr_scheduler_step(
  sched : LRScheduler,
  opt : OptimizerParams,
) -> (LRScheduler, OptimizerParams) {
  match sched {
    None => (sched, opt)
    StepLR(s) => {
      let lr = step_decay(s.base_lr, s.gamma, s.step_size, s.step)
      let opt2 = optimizer_params_set_lr(opt, lr)
      s.step = s.step + 1
      (StepLR(s), opt2)
    }
    CosineLR(s) => {
      let lr = cosine_decay(s.base_lr, s.min_lr, s.total_steps, s.step)
      let opt2 = optimizer_params_set_lr(opt, lr)
      s.step = s.step + 1
      (CosineLR(s), opt2)
    }
    WarmupCosineLR(s) => {
      let lr = warmup_cosine(
        s.base_lr,
        s.min_lr,
        s.total_steps,
        s.warmup_steps,
        s.step,
      )
      let opt2 = optimizer_params_set_lr(opt, lr)
      s.step = s.step + 1
      (WarmupCosineLR(s), opt2)
    }
  }
}

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
fn metrics_none(
  _logits : @math.Tensor,
  _labels : Array[Int],
) -> Map[String, Double] {
  Map::new()
}

///|
pub fn train_step_with_metrics(
  model : Sequential,
  x : AutoTensor,
  y_true : Array[Int],
  opt : OptimizerParams,
  metrics_fn? : (@math.Tensor, Array[Int]) -> Map[String, Double] = metrics_none,
) -> (Double, OptimizerParams, Map[String, Double]) {
  model.train()
  let opt2 = optimizer_params_zero_grad(opt)
  let (logits, ctx) = model.forward(x)
  let loss = logits.softmax_cross_entropy(y_true)
  loss.backward()
  model.sync_grads(ctx)
  let opt3 = optimizer_params_step(opt2)
  let metrics = metrics_fn(logits.value(), y_true)
  (loss.value().get([0]), opt3, metrics)
}

///|
pub fn eval_step_with_metrics(
  model : Sequential,
  x : AutoTensor,
  y_true : Array[Int],
  metrics_fn? : (@math.Tensor, Array[Int]) -> Map[String, Double] = metrics_none,
) -> (Double, Map[String, Double]) {
  model.eval()
  let (logits, _ctx) = model.forward(x)
  let loss = logits.softmax_cross_entropy(y_true)
  let metrics = metrics_fn(logits.value(), y_true)
  (loss.value().get([0]), metrics)
}

///|
pub fn train_step_batch_with_metrics(
  model : Sequential,
  batch : @data.Batch,
  opt : OptimizerParams,
  metrics_fn? : (@math.Tensor, Array[Int]) -> Map[String, Double] = metrics_none,
) -> (Double, OptimizerParams, Map[String, Double]) {
  match batch.labels {
    @data.Single(labels) => {
      if batch.features.length() != labels.length() {
        abort("batch features 和 labels 的样本数不匹配")
      }
      let x = batch_features_to_autotensor(batch.features)
      let y_true = batch_labels_to_indices(labels)
      train_step_with_metrics(model, x, y_true, opt, metrics_fn~)
    }
    @data.Multi(_) =>
      abort("train_step_batch_with_metrics 仅支持单标签分类数据")
  }
}

///|
pub fn eval_step_batch_with_metrics(
  model : Sequential,
  batch : @data.Batch,
  metrics_fn? : (@math.Tensor, Array[Int]) -> Map[String, Double] = metrics_none,
) -> (Double, Map[String, Double]) {
  match batch.labels {
    @data.Single(labels) => {
      if batch.features.length() != labels.length() {
        abort("batch features 和 labels 的样本数不匹配")
      }
      let x = batch_features_to_autotensor(batch.features)
      let y_true = batch_labels_to_indices(labels)
      eval_step_with_metrics(model, x, y_true, metrics_fn~)
    }
    @data.Multi(_) =>
      abort("eval_step_batch_with_metrics 仅支持单标签分类数据")
  }
}

///|
pub fn train_epoch(
  model : Sequential,
  loader : @data.DataLoader,
  opt : OptimizerParams,
  epoch? : Int = 0,
  scheduler? : LRScheduler = None,
  callbacks? : TrainCallbacks = TrainCallbacks::none(),
  metrics_fn? : (@math.Tensor, Array[Int]) -> Map[String, Double] = metrics_none,
) -> (TrainMetrics, OptimizerParams, LRScheduler) {
  let metrics = TrainMetrics::new()
  let mut opt2 = opt
  let mut sched2 = scheduler
  let cb = callbacks
  let ep = epoch
  loader.reset()
  let mut batch_idx = 0
  while true {
    match loader.next_batch_pack() {
      Some(batch) => {
        let batch_size = batch.features.length()
        let (loss, opt3, extra) = train_step_batch_with_metrics(
          model,
          batch,
          opt2,
          metrics_fn~,
        )
        metrics.update_loss(loss, batch_size)
        metrics.add_metrics_avg(extra, batch_size)
        let (sched3, opt4) = lr_scheduler_step(sched2, opt3)
        opt2 = opt4
        sched2 = sched3
        (cb.on_batch_end)(ep, batch_idx, loss, metrics, opt2)
        batch_idx = batch_idx + 1
      }
      None => break
    }
  }
  (cb.on_epoch_end)(ep, metrics, opt2)
  (metrics, opt2, sched2)
}

///|
pub fn eval_epoch(
  model : Sequential,
  loader : @data.DataLoader,
  metrics_fn? : (@math.Tensor, Array[Int]) -> Map[String, Double] = metrics_none,
) -> TrainMetrics {
  let metrics = TrainMetrics::new()
  loader.reset()
  while true {
    match loader.next_batch_pack() {
      Some(batch) => {
        let batch_size = batch.features.length()
        let (loss, extra) = eval_step_batch_with_metrics(
          model,
          batch,
          metrics_fn~,
        )
        metrics.update_loss(loss, batch_size)
        metrics.add_metrics_avg(extra, batch_size)
      }
      None => break
    }
  }
  metrics
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
  let (loss, opt2, _metrics) = train_step_with_metrics(model, x, y_true, opt)
  (loss, opt2)
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
  let (loss, _metrics) = eval_step_with_metrics(model, x, y_true)
  loss
}
