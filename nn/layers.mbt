///|
/// 基础层与激活（前向版）

///|
pub struct Linear {
  mut w : Array[Array[Double]]
  mut b : Array[Double]
  in_features : Int
  out_features : Int
}

///|
pub fn Linear::new(in_features : Int, out_features : Int) -> Linear {
  if in_features <= 0 || out_features <= 0 {
    abort("in_features/out_features 必须大于 0")
  }
  let w = Array::make(out_features, [])
  for i = 0; i < out_features; i = i + 1 {
    w[i] = Array::make(in_features, 0.0)
  }
  let b = Array::make(out_features, 0.0)
  { w, b, in_features, out_features }
}

///|
/// 设置参数（用于测试或手动初始化）
pub fn Linear::set_params(
  self : Linear,
  w : Array[Array[Double]],
  b : Array[Double],
) -> Unit {
  if w.length() != self.out_features || b.length() != self.out_features {
    abort("参数尺寸不匹配")
  }
  for i = 0; i < w.length(); i = i + 1 {
    if w[i].length() != self.in_features {
      abort("权重维度不匹配")
    }
  }
  self.w = w
  self.b = b
}

///|
/// 前向：X (n_samples x in_features) -> (n_samples x out_features)
pub fn Linear::forward(self : Linear, x : @math.Matrix) -> @math.Matrix {
  let (n_samples, n_features) = x.shape()
  if n_features != self.in_features {
    abort("输入特征维度不匹配")
  }
  let out = @math.Matrix::zeros(n_samples, self.out_features)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    for o = 0; o < self.out_features; o = o + 1 {
      let mut sum = self.b[o]
      for f = 0; f < self.in_features; f = f + 1 {
        sum = sum + row.get(f) * self.w[o][f]
      }
      out.set(i, o, sum)
    }
  }
  out
}

///|
pub enum Activation {
  Relu
  Sigmoid
  Tanh
  Softmax
}

///|
pub fn act_relu() -> Activation {
  Relu
}

///|
pub fn act_sigmoid() -> Activation {
  Sigmoid
}

///|
pub fn act_tanh() -> Activation {
  Tanh
}

///|
pub fn act_softmax() -> Activation {
  Softmax
}

///|
fn tensor_to_matrix(t : @math.Tensor) -> @math.Matrix {
  let shape = t.get_shape()
  if shape.length() != 2 {
    abort("Tensor 需要形状 [batch, features]")
  }
  let rows = shape[0]
  let cols = shape[1]
  let data = Array::make(rows, [])
  for i = 0; i < rows; i = i + 1 {
    let row = Array::make(cols, 0.0)
    for j = 0; j < cols; j = j + 1 {
      row[j] = t.get([i, j])
    }
    data[i] = row
  }
  @math.Matrix::new(data)
}

///|
fn matrix_to_tensor(m : @math.Matrix) -> @math.Tensor {
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
fn apply_activation_scalar(x : Double, act : Activation) -> Double {
  match act {
    Relu => if x > 0.0 { x } else { 0.0 }
    Sigmoid => 1.0 / (1.0 + @moonbitlang/core/math.exp(-x))
    Tanh => @moonbitlang/core/math.tanh(x)
    Softmax => x // placeholder, real softmax handled separately
  }
}

///|
/// 对每个元素应用（Softmax 单独处理）
pub fn activation_forward(x : @math.Matrix, act : Activation) -> @math.Matrix {
  let (n, m) = x.shape()
  match act {
    Softmax => {
      let out = @math.Matrix::zeros(n, m)
      for i = 0; i < n; i = i + 1 {
        let row = x.get_row(i)
        let mut max_v = row.get(0)
        for j = 1; j < m; j = j + 1 {
          let v = row.get(j)
          if v > max_v {
            max_v = v
          }
        }
        let mut sum = 0.0
        for j = 0; j < m; j = j + 1 {
          let v = @moonbitlang/core/math.exp(row.get(j) - max_v)
          out.set(i, j, v)
          sum = sum + v
        }
        for j = 0; j < m; j = j + 1 {
          out.set(i, j, out.get(i, j) / sum)
        }
      }
      out
    }
    _ => {
      let out = @math.Matrix::zeros(n, m)
      for i = 0; i < n; i = i + 1 {
        let row = x.get_row(i)
        for j = 0; j < m; j = j + 1 {
          out.set(i, j, apply_activation_scalar(row.get(j), act))
        }
      }
      out
    }
  }
}

///|
pub fn Linear::forward_tensor(self : Linear, x : @math.Tensor) -> @math.Tensor {
  let mat = tensor_to_matrix(x)
  let out = self.forward(mat)
  matrix_to_tensor(out)
}

///|
pub fn activation_forward_tensor(
  x : @math.Tensor,
  act : Activation,
) -> @math.Tensor {
  let mat = tensor_to_matrix(x)
  let out = activation_forward(mat, act)
  matrix_to_tensor(out)
}

///|
/// 均匀随机初始化 [-scale, scale]
pub fn linear_rand_init(
  in_features : Int,
  out_features : Int,
  scale? : Double = 0.01,
  seed? : Int = 42,
) -> Linear {
  let lin = Linear::new(in_features, out_features)
  let mut rng = seed
  let w = Array::make(out_features, [])
  for o = 0; o < out_features; o = o + 1 {
    let row = Array::make(in_features, 0.0)
    for f = 0; f < in_features; f = f + 1 {
      rng = (1103515245 * rng + 12345) & 0x7fffffff
      let r = rng.to_double() / (0x7fffffff).to_double()
      row[f] = (r * 2.0 - 1.0) * scale
    }
    w[o] = row
  }
  let b = Array::make(out_features, 0.0)
  lin.set_params(w, b)
  lin
}
