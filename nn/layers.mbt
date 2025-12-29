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

///|
/// Dropout
pub struct Dropout {
  rate : Double
  mut seed : Int
}

///|
pub fn Dropout::new(rate : Double, seed? : Int = 42) -> Dropout {
  if rate < 0.0 || rate >= 1.0 {
    abort("rate 必须在 [0,1)")
  }
  { rate, seed }
}

///|
pub fn Dropout::forward(
  self : Dropout,
  x : @math.Matrix,
  train? : Bool = true,
) -> @math.Matrix {
  if not(train) || self.rate == 0.0 {
    return x
  }
  let (n, m) = x.shape()
  let out = @math.Matrix::zeros(n, m)
  let keep = 1.0 - self.rate
  let scale = 1.0 / keep
  let mut rng = self.seed
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      rng = (1103515245 * rng + 12345) & 0x7fffffff
      let r = rng.to_double() / (0x7fffffff).to_double()
      if r < keep {
        out.set(i, j, x.get(i, j) * scale)
      } else {
        out.set(i, j, 0.0)
      }
    }
  }
  self.seed = rng
  out
}

///|
/// BatchNorm1d
pub struct BatchNorm1d {
  gamma : Array[Double]
  beta : Array[Double]
  running_mean : Array[Double]
  running_var : Array[Double]
  momentum : Double
  eps : Double
}

///|
pub fn BatchNorm1d::new(
  num_features : Int,
  momentum? : Double = 0.9,
  eps? : Double = 1.0e-5,
) -> BatchNorm1d {
  if num_features <= 0 {
    abort("num_features 必须大于 0")
  }
  if momentum < 0.0 || momentum > 1.0 {
    abort("momentum 必须在 [0,1]")
  }
  let gamma = Array::make(num_features, 1.0)
  let beta = Array::make(num_features, 0.0)
  let running_mean = Array::make(num_features, 0.0)
  let running_var = Array::make(num_features, 1.0)
  { gamma, beta, running_mean, running_var, momentum, eps }
}

///|
pub fn BatchNorm1d::forward(
  self : BatchNorm1d,
  x : @math.Matrix,
  train? : Bool = true,
) -> @math.Matrix {
  let (out, _mean, _var) = self.forward_with_cache(x, train~)
  out
}

///|
pub fn BatchNorm1d::forward_with_cache(
  self : BatchNorm1d,
  x : @math.Matrix,
  train? : Bool = true,
) -> (@math.Matrix, Array[Double], Array[Double]) {
  let (n, m) = x.shape()
  if m != self.gamma.length() {
    abort("输入特征维度不匹配")
  }
  let mean = Array::make(m, 0.0)
  let variance = Array::make(m, 0.0)
  if train {
    for j = 0; j < m; j = j + 1 {
      let mut sum = 0.0
      for i = 0; i < n; i = i + 1 {
        sum = sum + x.get(i, j)
      }
      mean[j] = sum / n.to_double()
    }
    for j = 0; j < m; j = j + 1 {
      let mut sum = 0.0
      for i = 0; i < n; i = i + 1 {
        let diff = x.get(i, j) - mean[j]
        sum = sum + diff * diff
      }
      variance[j] = sum / n.to_double()
    }
    for j = 0; j < m; j = j + 1 {
      self.running_mean[j] = self.momentum * self.running_mean[j] +
        (1.0 - self.momentum) * mean[j]
      self.running_var[j] = self.momentum * self.running_var[j] +
        (1.0 - self.momentum) * variance[j]
    }
  } else {
    for j = 0; j < m; j = j + 1 {
      mean[j] = self.running_mean[j]
      variance[j] = self.running_var[j]
    }
  }
  let out = @math.Matrix::zeros(n, m)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      let norm = (x.get(i, j) - mean[j]) / (variance[j] + self.eps).sqrt()
      out.set(i, j, norm * self.gamma[j] + self.beta[j])
    }
  }
  (out, mean, variance)
}

///|
/// LayerNorm
pub struct LayerNorm {
  gamma : Array[Double]
  beta : Array[Double]
  eps : Double
}

///|
pub fn LayerNorm::new(num_features : Int, eps? : Double = 1.0e-5) -> LayerNorm {
  if num_features <= 0 {
    abort("num_features 必须大于 0")
  }
  {
    gamma: Array::make(num_features, 1.0),
    beta: Array::make(num_features, 0.0),
    eps,
  }
}

///|
pub fn LayerNorm::forward(self : LayerNorm, x : @math.Matrix) -> @math.Matrix {
  let (n, m) = x.shape()
  if m != self.gamma.length() {
    abort("输入特征维度不匹配")
  }
  let out = @math.Matrix::zeros(n, m)
  for i = 0; i < n; i = i + 1 {
    let mut mean = 0.0
    for j = 0; j < m; j = j + 1 {
      mean = mean + x.get(i, j)
    }
    mean = mean / m.to_double()
    let mut variance = 0.0
    for j = 0; j < m; j = j + 1 {
      let diff = x.get(i, j) - mean
      variance = variance + diff * diff
    }
    variance = variance / m.to_double()
    let denom = (variance + self.eps).sqrt()
    for j = 0; j < m; j = j + 1 {
      let norm = (x.get(i, j) - mean) / denom
      out.set(i, j, norm * self.gamma[j] + self.beta[j])
    }
  }
  out
}

///|
/// Embedding
pub struct Embedding {
  weight : Array[Array[Double]]
  num_embeddings : Int
  dim : Int
  seed : Int
}

///|
pub fn Embedding::new(
  num_embeddings : Int,
  dim : Int,
  seed? : Int = 42,
) -> Embedding {
  if num_embeddings <= 0 || dim <= 0 {
    abort("num_embeddings/dim 必须大于 0")
  }
  let mut rng = seed
  let weight = Array::make(num_embeddings, [])
  let scale = (1.0 / dim.to_double()).sqrt()
  for i = 0; i < num_embeddings; i = i + 1 {
    let row = Array::make(dim, 0.0)
    for j = 0; j < dim; j = j + 1 {
      rng = (1103515245 * rng + 12345) & 0x7fffffff
      let r = rng.to_double() / (0x7fffffff).to_double()
      row[j] = (r * 2.0 - 1.0) * scale
    }
    weight[i] = row
  }
  { weight, num_embeddings, dim, seed: rng }
}

///|
pub fn Embedding::forward(
  self : Embedding,
  indices : Array[Int],
) -> @math.Matrix {
  let n = indices.length()
  let out = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    let idx = indices[i]
    if idx < 0 || idx >= self.num_embeddings {
      abort("索引越界")
    }
    out[i] = self.weight[idx]
  }
  @math.Matrix::new(out)
}

///|
/// Conv2d
pub struct Conv2d {
  mut weight : @math.Tensor
  mut bias : @math.Tensor
  stride : Int
  padding : Int
}

///|
pub fn Conv2d::new(
  in_channels : Int,
  out_channels : Int,
  kernel_size : Int,
  stride? : Int = 1,
  padding? : Int = 0,
  seed? : Int = 42,
) -> Conv2d {
  if in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 {
    abort("in/out_channels 与 kernel_size 必须大于 0")
  }
  let mut rng = seed
  let scale = (1.0 / (in_channels * kernel_size * kernel_size).to_double()).sqrt()
  let total = out_channels * in_channels * kernel_size * kernel_size
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    rng = (1103515245 * rng + 12345) & 0x7fffffff
    let r = rng.to_double() / (0x7fffffff).to_double()
    data[i] = (r * 2.0 - 1.0) * scale
  }
  let weight = @math.Tensor::new(data, [
    out_channels, in_channels, kernel_size, kernel_size,
  ])
  let bias = @math.Tensor::zeros([out_channels])
  { weight, bias, stride, padding }
}

///|
pub fn Conv2d::set_params(
  self : Conv2d,
  weight : @math.Tensor,
  bias : @math.Tensor,
) -> Unit {
  if weight.get_shape().length() != 4 || bias.get_shape().length() != 1 {
    abort("conv2d 参数形状不匹配")
  }
  self.weight = weight
  self.bias = bias
}

///|
pub fn Conv2d::forward(self : Conv2d, x : @math.Tensor) -> @math.Tensor {
  let shape_x = x.get_shape()
  let shape_w = self.weight.get_shape()
  if shape_x.length() != 4 || shape_w.length() != 4 {
    abort("conv2d 仅支持 4D 张量")
  }
  let n = shape_x[0]
  let cin = shape_x[1]
  let h = shape_x[2]
  let wi = shape_x[3]
  let cout = shape_w[0]
  let cin_w = shape_w[1]
  let kh = shape_w[2]
  let kw = shape_w[3]
  if cin != cin_w {
    abort("conv2d 输入通道不匹配")
  }
  let out_h = (h + 2 * self.padding - kh) / self.stride + 1
  let out_w = (wi + 2 * self.padding - kw) / self.stride + 1
  let out = @math.Tensor::zeros([n, cout, out_h, out_w])
  for ni = 0; ni < n; ni = ni + 1 {
    for co = 0; co < cout; co = co + 1 {
      for oh = 0; oh < out_h; oh = oh + 1 {
        for ow = 0; ow < out_w; ow = ow + 1 {
          let mut sum = self.bias.get([co])
          for ci = 0; ci < cin; ci = ci + 1 {
            for khi = 0; khi < kh; khi = khi + 1 {
              for kwi = 0; kwi < kw; kwi = kwi + 1 {
                let ih = oh * self.stride + khi - self.padding
                let iw = ow * self.stride + kwi - self.padding
                if ih >= 0 && ih < h && iw >= 0 && iw < wi {
                  sum = sum +
                    x.get([ni, ci, ih, iw]) *
                    self.weight.get([co, ci, khi, kwi])
                }
              }
            }
          }
          out.set([ni, co, oh, ow], sum)
        }
      }
    }
  }
  out
}

///|
/// BatchNorm2d
pub struct BatchNorm2d {
  gamma : Array[Double]
  beta : Array[Double]
  running_mean : Array[Double]
  running_var : Array[Double]
  momentum : Double
  eps : Double
}

///|
pub fn BatchNorm2d::new(
  num_features : Int,
  momentum? : Double = 0.9,
  eps? : Double = 1.0e-5,
) -> BatchNorm2d {
  if num_features <= 0 {
    abort("num_features 必须大于 0")
  }
  if momentum < 0.0 || momentum > 1.0 {
    abort("momentum 必须在 [0,1]")
  }
  let gamma = Array::make(num_features, 1.0)
  let beta = Array::make(num_features, 0.0)
  let running_mean = Array::make(num_features, 0.0)
  let running_var = Array::make(num_features, 1.0)
  { gamma, beta, running_mean, running_var, momentum, eps }
}

///|
pub fn BatchNorm2d::forward(
  self : BatchNorm2d,
  x : @math.Tensor,
  train? : Bool = true,
) -> @math.Tensor {
  let shape = x.get_shape()
  if shape.length() != 4 {
    abort("batchnorm2d 仅支持 4D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  let h = shape[2]
  let w = shape[3]
  if c != self.gamma.length() {
    abort("输入通道不匹配")
  }
  let mean = Array::make(c, 0.0)
  let variance = Array::make(c, 0.0)
  if train {
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
    for ci = 0; ci < c; ci = ci + 1 {
      self.running_mean[ci] = self.momentum * self.running_mean[ci] +
        (1.0 - self.momentum) * mean[ci]
      self.running_var[ci] = self.momentum * self.running_var[ci] +
        (1.0 - self.momentum) * variance[ci]
    }
  } else {
    for ci = 0; ci < c; ci = ci + 1 {
      mean[ci] = self.running_mean[ci]
      variance[ci] = self.running_var[ci]
    }
  }
  let out = @math.Tensor::zeros([n, c, h, w])
  for ni = 0; ni < n; ni = ni + 1 {
    for ci = 0; ci < c; ci = ci + 1 {
      let inv_std = 1.0 / (variance[ci] + self.eps).sqrt()
      for hi = 0; hi < h; hi = hi + 1 {
        for wi = 0; wi < w; wi = wi + 1 {
          let norm = (x.get([ni, ci, hi, wi]) - mean[ci]) * inv_std
          out.set([ni, ci, hi, wi], norm * self.gamma[ci] + self.beta[ci])
        }
      }
    }
  }
  out
}

///|
/// Dropout2d
pub struct Dropout2d {
  rate : Double
  mut seed : Int
}

///|
pub fn Dropout2d::new(rate : Double, seed? : Int = 42) -> Dropout2d {
  if rate < 0.0 || rate >= 1.0 {
    abort("rate 必须在 [0,1)")
  }
  { rate, seed }
}

///|
pub fn Dropout2d::forward(
  self : Dropout2d,
  x : @math.Tensor,
  train? : Bool = true,
) -> @math.Tensor {
  if not(train) || self.rate == 0.0 {
    return x
  }
  let shape = x.get_shape()
  if shape.length() != 4 {
    abort("dropout2d 仅支持 4D 张量")
  }
  let n = shape[0]
  let c = shape[1]
  let h = shape[2]
  let w = shape[3]
  let out = @math.Tensor::zeros([n, c, h, w])
  let keep = 1.0 - self.rate
  let scale = 1.0 / keep
  let mut rng = self.seed
  for ni = 0; ni < n; ni = ni + 1 {
    for ci = 0; ci < c; ci = ci + 1 {
      rng = (1103515245 * rng + 12345) & 0x7fffffff
      let r = rng.to_double() / (0x7fffffff).to_double()
      let keep_channel = r < keep
      for hi = 0; hi < h; hi = hi + 1 {
        for wi = 0; wi < w; wi = wi + 1 {
          if keep_channel {
            out.set([ni, ci, hi, wi], x.get([ni, ci, hi, wi]) * scale)
          } else {
            out.set([ni, ci, hi, wi], 0.0)
          }
        }
      }
    }
  }
  self.seed = rng
  out
}
