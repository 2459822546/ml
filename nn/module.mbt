///|
/// 简易 Module/Parameter 体系（配合 AutoTensor）

///|
pub struct Parameter {
  mut data : @math.Tensor
  mut grad : @math.Tensor
  requires_grad : Bool
}

///|
fn module_matrix_to_tensor(m : @math.Matrix) -> @math.Tensor {
  let (rows, cols) = m.shape()
  let data = Array::make(rows * cols, 0.0)
  let mut idx = 0
  for i = 0; i < rows; i = i + 1 {
    for j = 0; j < cols; j = j + 1 {
      data[idx] = m.get(i, j)
      idx = idx + 1
    }
  }
  @math.Tensor::new(data, [rows, cols])
}

///|
fn module_array_to_tensor(values : Array[Double]) -> @math.Tensor {
  @math.Tensor::new(values, [values.length()])
}

///|
fn module_tensor_to_array(t : @math.Tensor) -> Array[Double] {
  let shape = t.get_shape()
  if shape.length() != 1 {
    abort("只支持 1D Tensor 转数组")
  }
  let n = shape[0]
  let out = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    out[i] = t.get([i])
  }
  out
}

///|
fn module_batchnorm_stats(x : @math.Tensor) -> (Array[Double], Array[Double]) {
  let shape = x.get_shape()
  if shape.length() != 2 {
    abort("batchnorm1d 仅支持 2D 张量")
  }
  let n = shape[0]
  let m = shape[1]
  let mean = Array::make(m, 0.0)
  let variance = Array::make(m, 0.0)
  for j = 0; j < m; j = j + 1 {
    let mut sum = 0.0
    for i = 0; i < n; i = i + 1 {
      sum = sum + x.get([i, j])
    }
    mean[j] = sum / n.to_double()
  }
  for j = 0; j < m; j = j + 1 {
    let mut sum = 0.0
    for i = 0; i < n; i = i + 1 {
      let diff = x.get([i, j]) - mean[j]
      sum = sum + diff * diff
    }
    variance[j] = sum / n.to_double()
  }
  (mean, variance)
}

///|
fn module_batchnorm_forward_tensor(
  x : @math.Tensor,
  gamma : @math.Tensor,
  beta : @math.Tensor,
  mean : Array[Double],
  variance : Array[Double],
  eps : Double,
) -> @math.Tensor {
  let shape = x.get_shape()
  if shape.length() != 2 {
    abort("batchnorm1d 仅支持 2D 张量")
  }
  let n = shape[0]
  let m = shape[1]
  let data = Array::make(n * m, 0.0)
  let mut idx = 0
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      let norm = (x.get([i, j]) - mean[j]) / (variance[j] + eps).sqrt()
      data[idx] = norm * gamma.get([j]) + beta.get([j])
      idx = idx + 1
    }
  }
  @math.Tensor::new(data, [n, m])
}

///|
fn module_batchnorm2d_stats(x : @math.Tensor) -> (Array[Double], Array[Double]) {
  let shape = x.get_shape()
  if shape.length() != 4 {
    abort("batchnorm2d 仅支持 4D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  let h = shape[2]
  let w = shape[3]
  let mean = Array::make(c, 0.0)
  let variance = Array::make(c, 0.0)
  for ci = 0; ci < c; ci = ci + 1 {
    let mut sum = 0.0
    for ni = 0; ni < n; ni = ni + 1 {
      for hi = 0; hi < h; hi = hi + 1 {
        for wi = 0; wi < w; wi = wi + 1 {
          sum = sum + x.get([ni, ci, hi, wi])
        }
      }
    }
    mean[ci] = sum / (n * h * w).to_double()
  }
  for ci = 0; ci < c; ci = ci + 1 {
    let mut sum = 0.0
    for ni = 0; ni < n; ni = ni + 1 {
      for hi = 0; hi < h; hi = hi + 1 {
        for wi = 0; wi < w; wi = wi + 1 {
          let diff = x.get([ni, ci, hi, wi]) - mean[ci]
          sum = sum + diff * diff
        }
      }
    }
    variance[ci] = sum / (n * h * w).to_double()
  }
  (mean, variance)
}

///|
fn module_batchnorm2d_forward_tensor(
  x : @math.Tensor,
  gamma : @math.Tensor,
  beta : @math.Tensor,
  mean : Array[Double],
  variance : Array[Double],
  eps : Double,
) -> @math.Tensor {
  let shape = x.get_shape()
  if shape.length() != 4 {
    abort("batchnorm2d 仅支持 4D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  let h = shape[2]
  let w = shape[3]
  let out = @math.Tensor::zeros([n, c, h, w])
  for ni = 0; ni < n; ni = ni + 1 {
    for ci = 0; ci < c; ci = ci + 1 {
      let inv_std = 1.0 / (variance[ci] + eps).sqrt()
      for hi = 0; hi < h; hi = hi + 1 {
        for wi = 0; wi < w; wi = wi + 1 {
          let norm = (x.get([ni, ci, hi, wi]) - mean[ci]) * inv_std
          out.set([ni, ci, hi, wi], norm * gamma.get([ci]) + beta.get([ci]))
        }
      }
    }
  }
  out
}

///|
pub fn Parameter::new(
  data : @math.Tensor,
  requires_grad? : Bool = true,
) -> Parameter {
  let grad = @math.Tensor::zeros(data.get_shape())
  { data, grad, requires_grad }
}

///|
pub fn Parameter::zero_grad(self : Parameter) -> Unit {
  self.grad = @math.Tensor::zeros(self.data.get_shape())
}

///|
pub fn Parameter::set_data(self : Parameter, data : @math.Tensor) -> Unit {
  self.data = data
  self.grad = @math.Tensor::zeros(self.data.get_shape())
}

///|
pub fn Parameter::set_data_inplace(
  self : Parameter,
  data : @math.Tensor,
) -> Unit {
  self.data = data
}

///|
pub fn Parameter::set_grad(self : Parameter, grad : @math.Tensor) -> Unit {
  self.grad = grad
}

///|
pub fn Parameter::get_data(self : Parameter) -> @math.Tensor {
  self.data
}

///|
pub fn Parameter::get_grad(self : Parameter) -> @math.Tensor {
  self.grad
}

///|
pub fn Parameter::requires_grad(self : Parameter) -> Bool {
  self.requires_grad
}

///|
pub struct LinearModule {
  weight : Parameter
  bias : Parameter
  in_features : Int
  out_features : Int
}

///|
pub fn LinearModule::new(
  in_features : Int,
  out_features : Int,
  seed? : Int = 42,
) -> LinearModule {
  if in_features <= 0 || out_features <= 0 {
    abort("in_features/out_features 必须大于 0")
  }
  let mut rng = seed
  let weight = Array::make(in_features * out_features, 0.0)
  let scale = (1.0 / in_features.to_double()).sqrt()
  for i = 0; i < weight.length(); i = i + 1 {
    rng = (1103515245 * rng + 12345) & 0x7fffffff
    let r = rng.to_double() / (0x7fffffff).to_double()
    weight[i] = (r * 2.0 - 1.0) * scale
  }
  let w = @math.Tensor::new(weight, [in_features, out_features])
  let b = @math.Tensor::zeros([out_features])
  {
    weight: Parameter::new(w),
    bias: Parameter::new(b),
    in_features,
    out_features,
  }
}

///|
pub fn LinearModule::set_params(
  self : LinearModule,
  weight : @math.Tensor,
  bias : @math.Tensor,
) -> Unit {
  let shape_w = weight.get_shape()
  let shape_b = bias.get_shape()
  if shape_w.length() != 2 || shape_b.length() != 1 {
    abort("参数形状不匹配")
  }
  if shape_w[0] != self.in_features || shape_w[1] != self.out_features {
    abort("权重维度不匹配")
  }
  if shape_b[0] != self.out_features {
    abort("偏置维度不匹配")
  }
  self.weight.set_data(weight)
  self.bias.set_data(bias)
}

///|
pub struct LinearContext {
  weight : AutoTensor
  bias : AutoTensor
}

///|
pub fn LinearModule::forward(
  self : LinearModule,
  x : AutoTensor,
) -> (AutoTensor, LinearContext) {
  let w = x.from_other(
    self.weight.get_data(),
    requires_grad=self.weight.requires_grad(),
  )
  let b = x.from_other(
    self.bias.get_data(),
    requires_grad=self.bias.requires_grad(),
  )
  let y = x.matmul(w).add_bias(b)
  (y, { weight: w, bias: b })
}

///|
pub fn LinearModule::sync_grads(
  self : LinearModule,
  ctx : LinearContext,
) -> Unit {
  self.weight.grad = ctx.weight.grad()
  self.bias.grad = ctx.bias.grad()
}

///|
pub fn LinearModule::parameters(self : LinearModule) -> Array[Parameter] {
  [self.weight, self.bias]
}

///|
pub struct BatchNorm1dModule {
  gamma : Parameter
  beta : Parameter
  mut running_mean : Array[Double]
  mut running_var : Array[Double]
  momentum : Double
  eps : Double
}

///|
pub fn BatchNorm1dModule::new(
  num_features : Int,
  momentum? : Double = 0.9,
  eps? : Double = 1.0e-5,
) -> BatchNorm1dModule {
  if num_features <= 0 {
    abort("num_features 必须大于 0")
  }
  let gamma = Parameter::new(
    @math.Tensor::new(Array::make(num_features, 1.0), [num_features]),
  )
  let beta = Parameter::new(
    @math.Tensor::new(Array::make(num_features, 0.0), [num_features]),
  )
  let running_mean = Array::make(num_features, 0.0)
  let running_var = Array::make(num_features, 1.0)
  { gamma, beta, running_mean, running_var, momentum, eps }
}

///|
pub fn BatchNorm1dModule::from_layer(bn : BatchNorm1d) -> BatchNorm1dModule {
  let gamma = Parameter::new(module_array_to_tensor(bn.gamma))
  let beta = Parameter::new(module_array_to_tensor(bn.beta))
  {
    gamma,
    beta,
    running_mean: bn.running_mean,
    running_var: bn.running_var,
    momentum: bn.momentum,
    eps: bn.eps,
  }
}

///|
pub struct BatchNorm1dContext {
  gamma : AutoTensor
  beta : AutoTensor
}

///|
pub fn BatchNorm1dModule::forward(
  self : BatchNorm1dModule,
  x : AutoTensor,
  train? : Bool = true,
) -> (AutoTensor, BatchNorm1dContext) {
  let x_val = x.value()
  let (mean, variance) = if train {
    let (m, v) = module_batchnorm_stats(x_val)
    for j = 0; j < self.running_mean.length(); j = j + 1 {
      self.running_mean[j] = self.momentum * self.running_mean[j] +
        (1.0 - self.momentum) * m[j]
      self.running_var[j] = self.momentum * self.running_var[j] +
        (1.0 - self.momentum) * v[j]
    }
    (m, v)
  } else {
    (self.running_mean, self.running_var)
  }
  let gamma = x.from_other(
    self.gamma.get_data(),
    requires_grad=self.gamma.requires_grad(),
  )
  let beta = x.from_other(
    self.beta.get_data(),
    requires_grad=self.beta.requires_grad(),
  )
  let out = module_batchnorm_forward_tensor(
    x_val,
    gamma.value(),
    beta.value(),
    mean,
    variance,
    self.eps,
  )
  let y = x.batchnorm1d_cached(
    out,
    module_array_to_tensor(mean),
    module_array_to_tensor(variance),
    gamma,
    beta,
    self.eps,
    train~,
  )
  (y, { gamma, beta })
}

///|
pub fn BatchNorm1dModule::sync_grads(
  self : BatchNorm1dModule,
  ctx : BatchNorm1dContext,
) -> Unit {
  self.gamma.set_grad(ctx.gamma.grad())
  self.beta.set_grad(ctx.beta.grad())
}

///|
pub fn BatchNorm1dModule::parameters(
  self : BatchNorm1dModule,
) -> Array[Parameter] {
  [self.gamma, self.beta]
}

///|
pub struct LayerNormModule {
  gamma : Parameter
  beta : Parameter
  eps : Double
}

///|
pub fn LayerNormModule::new(
  num_features : Int,
  eps? : Double = 1.0e-5,
) -> LayerNormModule {
  if num_features <= 0 {
    abort("num_features 必须大于 0")
  }
  let gamma = Parameter::new(
    @math.Tensor::new(Array::make(num_features, 1.0), [num_features]),
  )
  let beta = Parameter::new(
    @math.Tensor::new(Array::make(num_features, 0.0), [num_features]),
  )
  { gamma, beta, eps }
}

///|
pub fn LayerNormModule::from_layer(ln : LayerNorm) -> LayerNormModule {
  {
    gamma: Parameter::new(module_array_to_tensor(ln.gamma)),
    beta: Parameter::new(module_array_to_tensor(ln.beta)),
    eps: ln.eps,
  }
}

///|
pub struct LayerNormContext {
  gamma : AutoTensor
  beta : AutoTensor
}

///|
pub fn LayerNormModule::forward(
  self : LayerNormModule,
  x : AutoTensor,
) -> (AutoTensor, LayerNormContext) {
  let gamma = x.from_other(
    self.gamma.get_data(),
    requires_grad=self.gamma.requires_grad(),
  )
  let beta = x.from_other(
    self.beta.get_data(),
    requires_grad=self.beta.requires_grad(),
  )
  let y = x.layernorm(gamma, beta, self.eps)
  (y, { gamma, beta })
}

///|
pub fn LayerNormModule::sync_grads(
  self : LayerNormModule,
  ctx : LayerNormContext,
) -> Unit {
  self.gamma.set_grad(ctx.gamma.grad())
  self.beta.set_grad(ctx.beta.grad())
}

///|
pub fn LayerNormModule::parameters(self : LayerNormModule) -> Array[Parameter] {
  [self.gamma, self.beta]
}

///|
pub struct EmbeddingModule {
  weight : Parameter
  num_embeddings : Int
  dim : Int
}

///|
pub fn EmbeddingModule::new(
  num_embeddings : Int,
  dim : Int,
  seed? : Int = 42,
) -> EmbeddingModule {
  let emb = Embedding::new(num_embeddings, dim, seed~)
  EmbeddingModule::from_layer(emb)
}

///|
pub fn EmbeddingModule::from_layer(e : Embedding) -> EmbeddingModule {
  let weight = module_matrix_to_tensor(@math.Matrix::new(e.weight))
  {
    weight: Parameter::new(weight),
    num_embeddings: e.num_embeddings,
    dim: e.dim,
  }
}

///|
pub struct EmbeddingContext {
  weight : AutoTensor
}

///|
pub fn EmbeddingModule::forward(
  self : EmbeddingModule,
  x : AutoTensor,
) -> (AutoTensor, EmbeddingContext) {
  let weight = x.from_other(
    self.weight.get_data(),
    requires_grad=self.weight.requires_grad(),
  )
  let y = x.embedding(weight)
  (y, { weight, })
}

///|
pub fn EmbeddingModule::sync_grads(
  self : EmbeddingModule,
  ctx : EmbeddingContext,
) -> Unit {
  self.weight.set_grad(ctx.weight.grad())
}

///|
pub fn EmbeddingModule::parameters(self : EmbeddingModule) -> Array[Parameter] {
  [self.weight]
}

///|
pub struct Conv2dModule {
  weight : Parameter
  bias : Parameter
  stride : Int
  padding : Int
}

///|
pub struct Conv2dContext {
  weight : AutoTensor
  bias : AutoTensor
}

///|
pub fn Conv2dModule::new(
  in_channels : Int,
  out_channels : Int,
  kernel_size : Int,
  stride? : Int = 1,
  padding? : Int = 0,
  seed? : Int = 42,
) -> Conv2dModule {
  let conv = Conv2d::new(
    in_channels,
    out_channels,
    kernel_size,
    stride~,
    padding~,
    seed~,
  )
  {
    weight: Parameter::new(conv.weight),
    bias: Parameter::new(conv.bias),
    stride: conv.stride,
    padding: conv.padding,
  }
}

///|
pub fn Conv2dModule::set_params(
  self : Conv2dModule,
  weight : @math.Tensor,
  bias : @math.Tensor,
) -> Unit {
  let shape_w = weight.get_shape()
  let shape_b = bias.get_shape()
  if shape_w.length() != 4 || shape_b.length() != 1 {
    abort("conv2d 参数形状不匹配")
  }
  if shape_w[0] != shape_b[0] {
    abort("conv2d bias 维度不匹配")
  }
  self.weight.set_data(weight)
  self.bias.set_data(bias)
}

///|
pub fn Conv2dModule::forward(
  self : Conv2dModule,
  x : AutoTensor,
) -> (AutoTensor, Conv2dContext) {
  let w = x.from_other(
    self.weight.get_data(),
    requires_grad=self.weight.requires_grad(),
  )
  let b = x.from_other(
    self.bias.get_data(),
    requires_grad=self.bias.requires_grad(),
  )
  let y = x.conv2d(w, b, self.stride, self.padding)
  (y, { weight: w, bias: b })
}

///|
pub fn Conv2dModule::sync_grads(
  self : Conv2dModule,
  ctx : Conv2dContext,
) -> Unit {
  self.weight.set_grad(ctx.weight.grad())
  self.bias.set_grad(ctx.bias.grad())
}

///|
pub fn Conv2dModule::parameters(self : Conv2dModule) -> Array[Parameter] {
  [self.weight, self.bias]
}

///|
pub struct BatchNorm2dModule {
  gamma : Parameter
  beta : Parameter
  mut running_mean : Array[Double]
  mut running_var : Array[Double]
  momentum : Double
  eps : Double
}

///|
pub struct BatchNorm2dContext {
  gamma : AutoTensor
  beta : AutoTensor
}

///|
pub fn BatchNorm2dModule::new(
  num_features : Int,
  momentum? : Double = 0.9,
  eps? : Double = 1.0e-5,
) -> BatchNorm2dModule {
  if num_features <= 0 {
    abort("num_features 必须大于 0")
  }
  let gamma = Parameter::new(
    @math.Tensor::new(Array::make(num_features, 1.0), [num_features]),
  )
  let beta = Parameter::new(
    @math.Tensor::new(Array::make(num_features, 0.0), [num_features]),
  )
  let running_mean = Array::make(num_features, 0.0)
  let running_var = Array::make(num_features, 1.0)
  { gamma, beta, running_mean, running_var, momentum, eps }
}

///|
pub fn BatchNorm2dModule::from_layer(bn : BatchNorm2d) -> BatchNorm2dModule {
  let gamma = Parameter::new(module_array_to_tensor(bn.gamma))
  let beta = Parameter::new(module_array_to_tensor(bn.beta))
  {
    gamma,
    beta,
    running_mean: bn.running_mean,
    running_var: bn.running_var,
    momentum: bn.momentum,
    eps: bn.eps,
  }
}

///|
pub fn BatchNorm2dModule::forward(
  self : BatchNorm2dModule,
  x : AutoTensor,
  train? : Bool = true,
) -> (AutoTensor, BatchNorm2dContext) {
  let x_val = x.value()
  let (mean, variance) = if train {
    let (m, v) = module_batchnorm2d_stats(x_val)
    for j = 0; j < self.running_mean.length(); j = j + 1 {
      self.running_mean[j] = self.momentum * self.running_mean[j] +
        (1.0 - self.momentum) * m[j]
      self.running_var[j] = self.momentum * self.running_var[j] +
        (1.0 - self.momentum) * v[j]
    }
    (m, v)
  } else {
    (self.running_mean, self.running_var)
  }
  let gamma = x.from_other(
    self.gamma.get_data(),
    requires_grad=self.gamma.requires_grad(),
  )
  let beta = x.from_other(
    self.beta.get_data(),
    requires_grad=self.beta.requires_grad(),
  )
  let out = module_batchnorm2d_forward_tensor(
    x_val,
    gamma.value(),
    beta.value(),
    mean,
    variance,
    self.eps,
  )
  let y = x.batchnorm2d_cached(
    out,
    module_array_to_tensor(mean),
    module_array_to_tensor(variance),
    gamma,
    beta,
    self.eps,
    train~,
  )
  (y, { gamma, beta })
}

///|
pub fn BatchNorm2dModule::sync_grads(
  self : BatchNorm2dModule,
  ctx : BatchNorm2dContext,
) -> Unit {
  self.gamma.set_grad(ctx.gamma.grad())
  self.beta.set_grad(ctx.beta.grad())
}

///|
pub fn BatchNorm2dModule::parameters(
  self : BatchNorm2dModule,
) -> Array[Parameter] {
  [self.gamma, self.beta]
}

///|
pub struct MaxPool2dModule {
  kernel_size : Int
  stride : Int
}

///|
pub fn MaxPool2dModule::new(
  kernel_size : Int,
  stride? : Int = 0,
) -> MaxPool2dModule {
  let pool = MaxPool2d::new(kernel_size, stride~)
  { kernel_size: pool.kernel_size, stride: pool.stride }
}

///|
pub fn MaxPool2dModule::forward(
  self : MaxPool2dModule,
  x : AutoTensor,
) -> AutoTensor {
  x.maxpool2d(self.kernel_size, self.stride)
}

///|
pub struct AvgPool2dModule {
  kernel_size : Int
  stride : Int
}

///|
pub fn AvgPool2dModule::new(
  kernel_size : Int,
  stride? : Int = 0,
) -> AvgPool2dModule {
  let pool = AvgPool2d::new(kernel_size, stride~)
  { kernel_size: pool.kernel_size, stride: pool.stride }
}

///|
pub fn AvgPool2dModule::forward(
  self : AvgPool2dModule,
  x : AutoTensor,
) -> AutoTensor {
  x.avgpool2d(self.kernel_size, self.stride)
}

///|
pub struct Flatten2dModule {
  dummy : Int
}

///|
pub fn Flatten2dModule::new() -> Flatten2dModule {
  { dummy: 0 }
}

///|
pub fn Flatten2dModule::forward(
  _self : Flatten2dModule,
  x : AutoTensor,
) -> AutoTensor {
  x.flatten2d()
}

///|
pub enum Module {
  Linear(LinearModule)
  Relu
  Dropout(Dropout)
  Dropout2d(Dropout2d)
  BatchNorm1d(BatchNorm1dModule)
  BatchNorm2d(BatchNorm2dModule)
  LayerNorm(LayerNormModule)
  Embedding(EmbeddingModule)
  Conv2d(Conv2dModule)
  MaxPool2d(MaxPool2dModule)
  AvgPool2d(AvgPool2dModule)
  Flatten2d(Flatten2dModule)
  Sequential(Sequential)
  ModuleList(ModuleList)
}

///|
pub fn module_linear(m : LinearModule) -> Module {
  Linear(m)
}

///|
pub fn module_relu() -> Module {
  Relu
}

///|
pub fn module_dropout(d : Dropout) -> Module {
  Dropout(d)
}

///|
pub fn module_dropout2d(d : Dropout2d) -> Module {
  Dropout2d(d)
}

///|
pub fn module_batchnorm1d(bn : BatchNorm1d) -> Module {
  BatchNorm1d(BatchNorm1dModule::from_layer(bn))
}

///|
pub fn module_batchnorm2d(bn : BatchNorm2d) -> Module {
  BatchNorm2d(BatchNorm2dModule::from_layer(bn))
}

///|
pub fn module_layernorm(ln : LayerNorm) -> Module {
  LayerNorm(LayerNormModule::from_layer(ln))
}

///|
pub fn module_embedding(e : Embedding) -> Module {
  Embedding(EmbeddingModule::from_layer(e))
}

///|
pub fn module_embedding_module(e : EmbeddingModule) -> Module {
  Embedding(e)
}

///|
pub fn module_conv2d(m : Conv2dModule) -> Module {
  Conv2d(m)
}

///|
pub fn module_maxpool2d(m : MaxPool2dModule) -> Module {
  MaxPool2d(m)
}

///|
pub fn module_avgpool2d(m : AvgPool2dModule) -> Module {
  AvgPool2d(m)
}

///|
pub fn module_flatten2d(m : Flatten2dModule) -> Module {
  Flatten2d(m)
}

///|
pub fn module_sequential(s : Sequential) -> Module {
  Sequential(s)
}

///|
pub fn module_modulelist(m : ModuleList) -> Module {
  ModuleList(m)
}

///|
pub enum ModuleContext {
  None
  Linear(LinearContext)
  BatchNorm1d(BatchNorm1dContext)
  BatchNorm2d(BatchNorm2dContext)
  LayerNorm(LayerNormContext)
  Embedding(EmbeddingContext)
  Conv2d(Conv2dContext)
  Sequential(Array[ModuleContext])
}

///|
pub fn module_forward(
  mod : Module,
  x : AutoTensor,
  train? : Bool = true,
) -> (AutoTensor, ModuleContext) {
  match mod {
    Linear(m) => {
      let (y, ctx) = m.forward(x)
      (y, ModuleContext::Linear(ctx))
    }
    Relu => (x.relu(), ModuleContext::None)
    Dropout(d) => {
      let (y, new_seed) = x.dropout(d.rate, d.seed, train~)
      d.seed = new_seed
      (y, ModuleContext::None)
    }
    Dropout2d(d) => {
      let (y, new_seed) = x.dropout2d(d.rate, d.seed, train~)
      d.seed = new_seed
      (y, ModuleContext::None)
    }
    BatchNorm1d(bn) => {
      let (y, ctx) = bn.forward(x, train~)
      (y, ModuleContext::BatchNorm1d(ctx))
    }
    BatchNorm2d(bn) => {
      let (y, ctx) = bn.forward(x, train~)
      (y, ModuleContext::BatchNorm2d(ctx))
    }
    LayerNorm(ln) => {
      let (y, ctx) = ln.forward(x)
      (y, ModuleContext::LayerNorm(ctx))
    }
    Embedding(e) => {
      let (y, ctx) = e.forward(x)
      (y, ModuleContext::Embedding(ctx))
    }
    Conv2d(m) => {
      let (y, ctx) = m.forward(x)
      (y, ModuleContext::Conv2d(ctx))
    }
    MaxPool2d(m) => (m.forward(x), ModuleContext::None)
    AvgPool2d(m) => (m.forward(x), ModuleContext::None)
    Flatten2d(m) => (m.forward(x), ModuleContext::None)
    Sequential(s) => {
      let seq = s
      seq.train = train
      let (y, ctxs) = seq.forward(x)
      (y, ModuleContext::Sequential(ctxs))
    }
    ModuleList(_) => abort("ModuleList 不能直接 forward")
  }
}

///|
pub fn module_sync_grads(mod : Module, ctx : ModuleContext) -> Unit {
  match mod {
    Linear(m) =>
      match ctx {
        ModuleContext::Linear(lc) => m.sync_grads(lc)
        _ => ()
      }
    BatchNorm1d(m) =>
      match ctx {
        ModuleContext::BatchNorm1d(bc) => m.sync_grads(bc)
        _ => ()
      }
    BatchNorm2d(m) =>
      match ctx {
        ModuleContext::BatchNorm2d(bc) => m.sync_grads(bc)
        _ => ()
      }
    LayerNorm(m) =>
      match ctx {
        ModuleContext::LayerNorm(lc) => m.sync_grads(lc)
        _ => ()
      }
    Embedding(m) =>
      match ctx {
        ModuleContext::Embedding(ec) => m.sync_grads(ec)
        _ => ()
      }
    Conv2d(m) =>
      match ctx {
        ModuleContext::Conv2d(cc) => m.sync_grads(cc)
        _ => ()
      }
    Sequential(s) =>
      match ctx {
        ModuleContext::Sequential(c) => s.sync_grads(c)
        _ => ()
      }
    _ => ()
  }
}

///|
pub fn module_parameters(mod : Module) -> Array[Parameter] {
  match mod {
    Linear(m) => m.parameters()
    BatchNorm1d(m) => m.parameters()
    BatchNorm2d(m) => m.parameters()
    LayerNorm(m) => m.parameters()
    Embedding(m) => m.parameters()
    Conv2d(m) => m.parameters()
    Sequential(s) => s.parameters()
    ModuleList(m) => m.parameters()
    _ => []
  }
}

///|
pub fn module_train(mod : Module) -> Module {
  match mod {
    Sequential(s) => {
      let seq = s
      seq.train = true
      for i = 0; i < seq.modules.length(); i = i + 1 {
        seq.modules[i] = module_train(seq.modules[i])
      }
      Sequential(seq)
    }
    ModuleList(m) => {
      let ml = m
      ml.train = true
      for i = 0; i < ml.modules.length(); i = i + 1 {
        ml.modules[i] = module_train(ml.modules[i])
      }
      ModuleList(ml)
    }
    _ => mod
  }
}

///|
pub fn module_eval(mod : Module) -> Module {
  match mod {
    Sequential(s) => {
      let seq = s
      seq.train = false
      for i = 0; i < seq.modules.length(); i = i + 1 {
        seq.modules[i] = module_eval(seq.modules[i])
      }
      Sequential(seq)
    }
    ModuleList(m) => {
      let ml = m
      ml.train = false
      for i = 0; i < ml.modules.length(); i = i + 1 {
        ml.modules[i] = module_eval(ml.modules[i])
      }
      ModuleList(ml)
    }
    _ => mod
  }
}

///|
pub struct Sequential {
  modules : Array[Module]
  mut train : Bool
}

///|
pub fn Sequential::new(modules : Array[Module]) -> Sequential {
  { modules, train: true }
}

///|
pub fn Sequential::forward(
  self : Sequential,
  x : AutoTensor,
) -> (AutoTensor, Array[ModuleContext]) {
  let contexts = Array::make(self.modules.length(), ModuleContext::None)
  let mut out = x
  for i = 0; i < self.modules.length(); i = i + 1 {
    let (y, ctx) = module_forward(self.modules[i], out, train=self.train)
    contexts[i] = ctx
    out = y
  }
  (out, contexts)
}

///|
pub fn Sequential::sync_grads(
  self : Sequential,
  contexts : Array[ModuleContext],
) -> Unit {
  if contexts.length() != self.modules.length() {
    abort("context 数量不匹配")
  }
  for i = 0; i < self.modules.length(); i = i + 1 {
    module_sync_grads(self.modules[i], contexts[i])
  }
}

///|
pub fn Sequential::parameters(self : Sequential) -> Array[Parameter] {
  let params = []
  for i = 0; i < self.modules.length(); i = i + 1 {
    let p = module_parameters(self.modules[i])
    for j = 0; j < p.length(); j = j + 1 {
      params.push(p[j])
    }
  }
  let out = Array::make(
    params.length(),
    Parameter::new(@math.Tensor::zeros([1])),
  )
  for i = 0; i < params.length(); i = i + 1 {
    out[i] = params[i]
  }
  out
}

///|
pub fn Sequential::train(self : Sequential) -> Unit {
  self.train = true
  for i = 0; i < self.modules.length(); i = i + 1 {
    self.modules[i] = module_train(self.modules[i])
  }
}

///|
pub fn Sequential::eval(self : Sequential) -> Unit {
  self.train = false
  for i = 0; i < self.modules.length(); i = i + 1 {
    self.modules[i] = module_eval(self.modules[i])
  }
}

///|
pub struct ModuleList {
  modules : Array[Module]
  mut train : Bool
}

///|
pub fn ModuleList::new(modules : Array[Module]) -> ModuleList {
  { modules, train: true }
}

///|
pub fn ModuleList::parameters(self : ModuleList) -> Array[Parameter] {
  let params = []
  for i = 0; i < self.modules.length(); i = i + 1 {
    let p = module_parameters(self.modules[i])
    for j = 0; j < p.length(); j = j + 1 {
      params.push(p[j])
    }
  }
  let out = Array::make(
    params.length(),
    Parameter::new(@math.Tensor::zeros([1])),
  )
  for i = 0; i < params.length(); i = i + 1 {
    out[i] = params[i]
  }
  out
}

///|
pub fn ModuleList::train(self : ModuleList) -> Unit {
  self.train = true
  for i = 0; i < self.modules.length(); i = i + 1 {
    self.modules[i] = module_train(self.modules[i])
  }
}

///|
pub fn ModuleList::eval(self : ModuleList) -> Unit {
  self.train = false
  for i = 0; i < self.modules.length(); i = i + 1 {
    self.modules[i] = module_eval(self.modules[i])
  }
}

///|
fn module_state_dict_key(prefix : String, name : String) -> String {
  if prefix == "" {
    name
  } else {
    prefix + name
  }
}

///|
fn module_state_dict_add(
  dict : Map[String, @math.Tensor],
  key : String,
  value : @math.Tensor,
) -> Unit {
  dict.set(key, value)
}

///|
fn module_state_dict_into(
  mod : Module,
  prefix : String,
  dict : Map[String, @math.Tensor],
) -> Unit {
  match mod {
    Linear(m) => {
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "weight"),
        m.weight.get_data(),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "bias"),
        m.bias.get_data(),
      )
    }
    BatchNorm1d(m) => {
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "gamma"),
        m.gamma.get_data(),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "beta"),
        m.beta.get_data(),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "running_mean"),
        module_array_to_tensor(m.running_mean),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "running_var"),
        module_array_to_tensor(m.running_var),
      )
    }
    BatchNorm2d(m) => {
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "gamma"),
        m.gamma.get_data(),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "beta"),
        m.beta.get_data(),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "running_mean"),
        module_array_to_tensor(m.running_mean),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "running_var"),
        module_array_to_tensor(m.running_var),
      )
    }
    LayerNorm(m) => {
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "gamma"),
        m.gamma.get_data(),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "beta"),
        m.beta.get_data(),
      )
    }
    Embedding(m) =>
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "weight"),
        m.weight.get_data(),
      )
    Conv2d(m) => {
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "weight"),
        m.weight.get_data(),
      )
      module_state_dict_add(
        dict,
        module_state_dict_key(prefix, "bias"),
        m.bias.get_data(),
      )
    }
    Sequential(s) =>
      for i = 0; i < s.modules.length(); i = i + 1 {
        let sub_prefix = prefix + i.to_string() + "."
        module_state_dict_into(s.modules[i], sub_prefix, dict)
      }
    ModuleList(m) =>
      for i = 0; i < m.modules.length(); i = i + 1 {
        let sub_prefix = prefix + i.to_string() + "."
        module_state_dict_into(m.modules[i], sub_prefix, dict)
      }
    _ => ()
  }
}

///|
pub fn module_state_dict(
  mod : Module,
  prefix? : String = "",
) -> Map[String, @math.Tensor] {
  let dict = Map::new()
  module_state_dict_into(mod, prefix, dict)
  dict
}

///|
fn module_state_dict_get(
  dict : Map[String, @math.Tensor],
  key : String,
  strict : Bool,
) -> @math.Tensor? {
  match dict.get(key) {
    Some(v) => Some(v)
    None => if strict { abort("state_dict 缺少键: " + key) } else { None }
  }
}

///|
pub fn module_load_state_dict(
  mod : Module,
  dict : Map[String, @math.Tensor],
  strict? : Bool = true,
  prefix? : String = "",
) -> Unit {
  let strict_val = strict
  match mod {
    Linear(m) => {
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "weight"),
          strict_val,
        ) {
        Some(v) => m.weight.set_data(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "bias"),
          strict_val,
        ) {
        Some(v) => m.bias.set_data(v)
        None => ()
      }
    }
    BatchNorm1d(m) => {
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "gamma"),
          strict_val,
        ) {
        Some(v) => m.gamma.set_data(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "beta"),
          strict_val,
        ) {
        Some(v) => m.beta.set_data(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "running_mean"),
          strict_val,
        ) {
        Some(v) => m.running_mean = module_tensor_to_array(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "running_var"),
          strict_val,
        ) {
        Some(v) => m.running_var = module_tensor_to_array(v)
        None => ()
      }
    }
    BatchNorm2d(m) => {
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "gamma"),
          strict_val,
        ) {
        Some(v) => m.gamma.set_data(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "beta"),
          strict_val,
        ) {
        Some(v) => m.beta.set_data(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "running_mean"),
          strict_val,
        ) {
        Some(v) => m.running_mean = module_tensor_to_array(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "running_var"),
          strict_val,
        ) {
        Some(v) => m.running_var = module_tensor_to_array(v)
        None => ()
      }
    }
    LayerNorm(m) => {
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "gamma"),
          strict_val,
        ) {
        Some(v) => m.gamma.set_data(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "beta"),
          strict_val,
        ) {
        Some(v) => m.beta.set_data(v)
        None => ()
      }
    }
    Embedding(m) =>
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "weight"),
          strict_val,
        ) {
        Some(v) => m.weight.set_data(v)
        None => ()
      }
    Conv2d(m) => {
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "weight"),
          strict_val,
        ) {
        Some(v) => m.weight.set_data(v)
        None => ()
      }
      match
        module_state_dict_get(
          dict,
          module_state_dict_key(prefix, "bias"),
          strict_val,
        ) {
        Some(v) => m.bias.set_data(v)
        None => ()
      }
    }
    Sequential(s) =>
      for i = 0; i < s.modules.length(); i = i + 1 {
        let sub_prefix = prefix + i.to_string() + "."
        module_load_state_dict(
          s.modules[i],
          dict,
          strict=strict_val,
          prefix=sub_prefix,
        )
      }
    ModuleList(m) =>
      for i = 0; i < m.modules.length(); i = i + 1 {
        let sub_prefix = prefix + i.to_string() + "."
        module_load_state_dict(
          m.modules[i],
          dict,
          strict=strict_val,
          prefix=sub_prefix,
        )
      }
    _ => ()
  }
}

///|
pub fn Sequential::state_dict(self : Sequential) -> Map[String, @math.Tensor] {
  module_state_dict(Sequential(self))
}

///|
pub fn Sequential::load_state_dict(
  self : Sequential,
  dict : Map[String, @math.Tensor],
  strict? : Bool = true,
) -> Unit {
  module_load_state_dict(Sequential(self), dict, strict~)
}

///|
pub fn ModuleList::state_dict(self : ModuleList) -> Map[String, @math.Tensor] {
  module_state_dict(ModuleList(self))
}

///|
pub fn ModuleList::load_state_dict(
  self : ModuleList,
  dict : Map[String, @math.Tensor],
  strict? : Bool = true,
) -> Unit {
  module_load_state_dict(ModuleList(self), dict, strict~)
}
