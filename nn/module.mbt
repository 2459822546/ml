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
pub enum Module {
  Linear(LinearModule)
  Relu
  Dropout(Dropout)
  BatchNorm1d(BatchNorm1d)
  LayerNorm(LayerNorm)
  Embedding(Embedding)
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
pub fn module_batchnorm1d(bn : BatchNorm1d) -> Module {
  BatchNorm1d(bn)
}

///|
pub fn module_layernorm(ln : LayerNorm) -> Module {
  LayerNorm(ln)
}

///|
pub fn module_embedding(e : Embedding) -> Module {
  Embedding(e)
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
    BatchNorm1d(bn) => {
      let (out, mean, variance) = bn.forward_with_cache(
        x.value().to_matrix(),
        train~,
      )
      let y = x.batchnorm1d_cached(
        module_matrix_to_tensor(out),
        module_array_to_tensor(mean),
        module_array_to_tensor(variance),
        module_array_to_tensor(bn.gamma),
        module_array_to_tensor(bn.beta),
        bn.eps,
        train~,
      )
      (y, ModuleContext::None)
    }
    LayerNorm(ln) => {
      let y = x.layernorm(
        module_array_to_tensor(ln.gamma),
        module_array_to_tensor(ln.beta),
        ln.eps,
      )
      (y, ModuleContext::None)
    }
    Embedding(e) => {
      let v = x.value().to_vector().to_array()
      let indices = Array::make(v.length(), 0)
      for i = 0; i < v.length(); i = i + 1 {
        indices[i] = v[i].to_int()
      }
      let out = e.forward(indices)
      (
        AutoTensor::from_other(
          x,
          module_matrix_to_tensor(out),
          requires_grad=false,
        ),
        ModuleContext::None,
      )
    }
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
