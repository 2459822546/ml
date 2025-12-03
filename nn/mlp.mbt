///|
/// 简单 MLP（前向版，手写参数更新）

///|
fn flatten_params(mat : Array[Array[Double]]) -> Array[Double] {
  let rows = mat.length()
  if rows == 0 {
    return []
  }
  let cols = mat[0].length()
  let out = Array::make(rows * cols, 0.0)
  let mut idx = 0
  for i = 0; i < rows; i = i + 1 {
    for j = 0; j < cols; j = j + 1 {
      out[idx] = mat[i][j]
      idx = idx + 1
    }
  }
  out
}

///|
fn to_matrix(
  arr : Array[Double],
  rows : Int,
  cols : Int,
) -> Array[Array[Double]] {
  if arr.length() != rows * cols {
    abort("尺寸不匹配")
  }
  let out = Array::make(rows, [])
  let mut idx = 0
  for i = 0; i < rows; i = i + 1 {
    let row = Array::make(cols, 0.0)
    for j = 0; j < cols; j = j + 1 {
      row[j] = arr[idx]
      idx = idx + 1
    }
    out[i] = row
  }
  out
}

///|
pub struct MLP {
  layers : Array[Linear]
  activations : Array[Activation]
  output_activation : Activation
}

///|
pub fn MLP::new(
  input_dim : Int,
  hidden : Array[Int],
  output_dim : Int,
  output_activation? : Activation = Softmax,
  seed? : Int = 123,
) -> MLP {
  if input_dim <= 0 || output_dim <= 0 {
    abort("输入/输出维度必须大于 0")
  }
  if hidden.length() == 0 {
    abort("至少需要一层隐藏层")
  }
  let dims = Array::make(hidden.length() + 2, 0)
  dims[0] = input_dim
  for i = 0; i < hidden.length(); i = i + 1 {
    if hidden[i] <= 0 {
      abort("隐藏层维度必须大于 0")
    }
    dims[i + 1] = hidden[i]
  }
  dims[dims.length() - 1] = output_dim
  let mut rng = seed
  let layers = Array::make(dims.length() - 1, Linear::new(1, 1))
  let acts = Array::make(dims.length() - 2, Relu)
  for i = 0; i < dims.length() - 1; i = i + 1 {
    let fan_in = dims[i]
    let fan_out = dims[i + 1]
    let scale = (2.0 / fan_in.to_double()).sqrt()
    let lin = linear_rand_init(fan_in, fan_out, scale~, seed=rng)
    rng = rng + 17
    layers[i] = lin
    if i < acts.length() {
      acts[i] = Relu
    }
  }
  { layers, activations: acts, output_activation }
}

///|

///|
pub fn MLP::forward(self : MLP, x : @math.Matrix) -> @math.Matrix {
  let mut out = x
  for i = 0; i < self.layers.length(); i = i + 1 {
    let lin = self.layers[i]
    let z = lin.forward(out)
    if i == self.layers.length() - 1 {
      out = activation_forward(z, self.output_activation)
    } else {
      out = activation_forward(z, self.activations[i])
    }
  }
  out
}

///|
pub fn MLP::mlp_predict_proba(self : MLP, x : @math.Matrix) -> @math.Matrix {
  self.forward(x)
}

///|
pub fn MLP::mlp_predict(self : MLP, x : @math.Matrix) -> Array[Int] {
  let prob = self.forward(x)
  let (n, c) = prob.shape()
  let labels = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    let row = prob.get_row(i)
    let mut best = 0
    let mut best_v = row.get(0)
    for j = 1; j < c; j = j + 1 {
      let v = row.get(j)
      if v > best_v {
        best_v = v
        best = j
      }
    }
    labels[i] = best
  }
  labels
}

///|
/// 简化的参数提取：只用于测试或手动更新
pub fn MLP::mlp_params(self : MLP) -> Array[OptimParam] {
  let params = Array::make(self.layers.length() * 2, OptimParam::new([]))
  let mut idx = 0
  for i = 0; i < self.layers.length(); i = i + 1 {
    let lin = self.layers[i]
    params[idx] = OptimParam::new(flatten_params(lin.w))
    idx = idx + 1
    params[idx] = OptimParam::new(lin.b)
    idx = idx + 1
  }
  params
}

///|
/// 将一维数组写回 Linear 参数（仅测试用）
pub fn MLP::mlp_set_params(self : MLP, params : Array[OptimParam]) -> Unit {
  if params.length() != self.layers.length() * 2 {
    abort("参数数量不匹配")
  }
  let mut idx = 0
  for i = 0; i < self.layers.length(); i = i + 1 {
    let w_vec = params[idx].data
    idx = idx + 1
    let b_vec = params[idx].data
    idx = idx + 1
    let lin = self.layers[i]
    let w = to_matrix(w_vec, lin.out_features, lin.in_features)
    if b_vec.length() != lin.out_features {
      abort("偏置长度不匹配")
    }
    lin.set_params(w, b_vec)
    self.layers[i] = lin
  }
}
