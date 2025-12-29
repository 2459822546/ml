///|
/// 简易计算图（Tensor 前向 + 反向传播）

///|
fn compute_strides(shape : Array[Int]) -> Array[Int] {
  let strides = Array::make(shape.length(), 0)
  let mut stride = 1
  for i = shape.length() - 1; i >= 0; i = i - 1 {
    strides[i] = stride
    stride = stride * shape[i]
  }
  strides
}

///|
fn indices_from_flat(
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
fn tensor_add_inplace(dst : @math.Tensor, src : @math.Tensor) -> Unit {
  let shape = dst.get_shape()
  if shape != src.get_shape() {
    abort("张量形状不一致")
  }
  let strides = compute_strides(shape)
  let total = dst.total_size()
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    dst.set(idx, dst.get(idx) + src.get(idx))
  }
}

///|
fn tensor_hadamard(a : @math.Tensor, b : @math.Tensor) -> @math.Tensor {
  let shape = a.get_shape()
  if shape != b.get_shape() {
    abort("张量形状不一致")
  }
  let strides = compute_strides(shape)
  let total = a.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = a.get(idx) * b.get(idx)
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_add(a : @math.Tensor, b : @math.Tensor) -> @math.Tensor {
  let shape = a.get_shape()
  if shape != b.get_shape() {
    abort("张量形状不一致")
  }
  let strides = compute_strides(shape)
  let total = a.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = a.get(idx) + b.get(idx)
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_neg(t : @math.Tensor) -> @math.Tensor {
  let shape = t.get_shape()
  let strides = compute_strides(shape)
  let total = t.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = -t.get(idx)
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_mask_relu(t : @math.Tensor) -> @math.Tensor {
  let shape = t.get_shape()
  let strides = compute_strides(shape)
  let total = t.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    let v = t.get(idx)
    data[i] = if v > 0.0 { 1.0 } else { 0.0 }
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_ones_like(t : @math.Tensor) -> @math.Tensor {
  @math.Tensor::fill(t.get_shape(), 1.0)
}

///|
fn tensor_zeros_like(t : @math.Tensor) -> @math.Tensor {
  @math.Tensor::zeros(t.get_shape())
}

///|
fn tensor_scale(t : @math.Tensor, scalar : Double) -> @math.Tensor {
  let shape = t.get_shape()
  let strides = compute_strides(shape)
  let total = t.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    data[i] = t.get(idx) * scalar
  }
  @math.Tensor::new(data, shape)
}

///|
fn tensor_matmul(a : @math.Tensor, b : @math.Tensor) -> @math.Tensor {
  let shape_a = a.get_shape()
  let shape_b = b.get_shape()
  if shape_a.length() != 2 || shape_b.length() != 2 {
    abort("matmul 仅支持 2D 张量")
  }
  let m = shape_a[0]
  let k = shape_a[1]
  let kb = shape_b[0]
  let n = shape_b[1]
  if k != kb {
    abort("matmul 维度不匹配")
  }
  let data = Array::make(m * n, 0.0)
  let mut idx = 0
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let mut sum = 0.0
      for p = 0; p < k; p = p + 1 {
        sum = sum + a.get([i, p]) * b.get([p, j])
      }
      data[idx] = sum
      idx = idx + 1
    }
  }
  @math.Tensor::new(data, [m, n])
}

///|
fn tensor_add_bias(x : @math.Tensor, bias : @math.Tensor) -> @math.Tensor {
  let shape_x = x.get_shape()
  let shape_b = bias.get_shape()
  if shape_x.length() != 2 || shape_b.length() != 1 {
    abort("add_bias 需要形状 [batch, features] + [features]")
  }
  let n = shape_x[0]
  let m = shape_x[1]
  if shape_b[0] != m {
    abort("bias 维度不匹配")
  }
  let data = Array::make(n * m, 0.0)
  let mut idx = 0
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      data[idx] = x.get([i, j]) + bias.get([j])
      idx = idx + 1
    }
  }
  @math.Tensor::new(data, [n, m])
}

///|
fn tensor_sum_all(x : @math.Tensor) -> @math.Tensor {
  let shape = x.get_shape()
  let strides = compute_strides(shape)
  let total = x.total_size()
  let mut sum = 0.0
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    sum = sum + x.get(idx)
  }
  @math.Tensor::new([sum], [1])
}

///|
pub enum Op {
  Input
  Add
  Sub
  Mul
  Relu
  MatMul
  AddBias
  Sum
}

///|
pub struct Node {
  value : @math.Tensor
  mut grad : @math.Tensor
  op : Op
  parents : Array[Int]
  requires_grad : Bool
}

///|
pub struct ComputationGraph {
  nodes : Array[Node]
}

///|
pub fn ComputationGraph::new() -> ComputationGraph {
  { nodes: [] }
}

///|
pub fn ComputationGraph::add_input(
  self : ComputationGraph,
  value : @math.Tensor,
  requires_grad? : Bool = true,
) -> Int {
  let grad = tensor_zeros_like(value)
  self.nodes.push({ value, grad, op: Input, parents: [], requires_grad })
  self.nodes.length() - 1
}

///|
pub fn ComputationGraph::add(self : ComputationGraph, a : Int, b : Int) -> Int {
  let va = self.nodes[a].value
  let vb = self.nodes[b].value
  let out = tensor_add(va, vb)
  let grad = tensor_zeros_like(out)
  self.nodes.push({
    value: out,
    grad,
    op: Add,
    parents: [a, b],
    requires_grad: true,
  })
  self.nodes.length() - 1
}

///|
pub fn ComputationGraph::sub(self : ComputationGraph, a : Int, b : Int) -> Int {
  let va = self.nodes[a].value
  let vb = self.nodes[b].value
  let out = tensor_add(va, tensor_neg(vb))
  let grad = tensor_zeros_like(out)
  self.nodes.push({
    value: out,
    grad,
    op: Sub,
    parents: [a, b],
    requires_grad: true,
  })
  self.nodes.length() - 1
}

///|
pub fn ComputationGraph::mul(self : ComputationGraph, a : Int, b : Int) -> Int {
  let va = self.nodes[a].value
  let vb = self.nodes[b].value
  let out = tensor_hadamard(va, vb)
  let grad = tensor_zeros_like(out)
  self.nodes.push({
    value: out,
    grad,
    op: Mul,
    parents: [a, b],
    requires_grad: true,
  })
  self.nodes.length() - 1
}

///|
pub fn ComputationGraph::relu(self : ComputationGraph, a : Int) -> Int {
  let va = self.nodes[a].value
  let shape = va.get_shape()
  let strides = compute_strides(shape)
  let total = va.total_size()
  let data = Array::make(total, 0.0)
  for i = 0; i < total; i = i + 1 {
    let idx = indices_from_flat(shape, strides, i)
    let v = va.get(idx)
    data[i] = if v > 0.0 { v } else { 0.0 }
  }
  let out = @math.Tensor::new(data, shape)
  let grad = tensor_zeros_like(out)
  self.nodes.push({
    value: out,
    grad,
    op: Relu,
    parents: [a],
    requires_grad: true,
  })
  self.nodes.length() - 1
}

///|
pub fn ComputationGraph::matmul(
  self : ComputationGraph,
  a : Int,
  b : Int,
) -> Int {
  let va = self.nodes[a].value
  let vb = self.nodes[b].value
  let out = tensor_matmul(va, vb)
  let grad = tensor_zeros_like(out)
  self.nodes.push({
    value: out,
    grad,
    op: MatMul,
    parents: [a, b],
    requires_grad: true,
  })
  self.nodes.length() - 1
}

///|
pub fn ComputationGraph::add_bias(
  self : ComputationGraph,
  x : Int,
  bias : Int,
) -> Int {
  let vx = self.nodes[x].value
  let vb = self.nodes[bias].value
  let out = tensor_add_bias(vx, vb)
  let grad = tensor_zeros_like(out)
  self.nodes.push({
    value: out,
    grad,
    op: AddBias,
    parents: [x, bias],
    requires_grad: true,
  })
  self.nodes.length() - 1
}

///|
pub fn ComputationGraph::sum(self : ComputationGraph, x : Int) -> Int {
  let vx = self.nodes[x].value
  let out = tensor_sum_all(vx)
  let grad = tensor_zeros_like(out)
  self.nodes.push({
    value: out,
    grad,
    op: Sum,
    parents: [x],
    requires_grad: true,
  })
  self.nodes.length() - 1
}

///|
fn topo_collect(
  g : ComputationGraph,
  root : Int,
  visited : Array[Bool],
  order : Array[Int],
) -> Unit {
  if visited[root] {
    return
  }
  visited[root] = true
  let parents = g.nodes[root].parents
  for i = 0; i < parents.length(); i = i + 1 {
    topo_collect(g, parents[i], visited, order)
  }
  order.push(root)
}

///|
pub fn ComputationGraph::backward(self : ComputationGraph, root : Int) -> Unit {
  self.backward_with_grad(root, tensor_ones_like(self.nodes[root].value))
}

///|
pub fn ComputationGraph::backward_with_grad(
  self : ComputationGraph,
  root : Int,
  grad_root : @math.Tensor,
) -> Unit {
  if root < 0 || root >= self.nodes.length() {
    abort("root 超出范围")
  }
  // 清零梯度
  for i = 0; i < self.nodes.length(); i = i + 1 {
    self.nodes[i].grad = tensor_zeros_like(self.nodes[i].value)
  }
  self.nodes[root].grad = grad_root
  let visited = Array::make(self.nodes.length(), false)
  let order = []
  topo_collect(self, root, visited, order)
  // 反向传播
  for idx = order.length() - 1; idx >= 0; idx = idx - 1 {
    let node_id = order[idx]
    let node = self.nodes[node_id]
    if not(node.requires_grad) {
      continue
    }
    match node.op {
      Input => ()
      Add => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        tensor_add_inplace(self.nodes[p0].grad, g)
        tensor_add_inplace(self.nodes[p1].grad, g)
      }
      Sub => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        tensor_add_inplace(self.nodes[p0].grad, g)
        tensor_add_inplace(self.nodes[p1].grad, tensor_neg(g))
      }
      Mul => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let v0 = self.nodes[p0].value
        let v1 = self.nodes[p1].value
        let grad0 = tensor_hadamard(g, v1)
        let grad1 = tensor_hadamard(g, v0)
        tensor_add_inplace(self.nodes[p0].grad, grad0)
        tensor_add_inplace(self.nodes[p1].grad, grad1)
      }
      Relu => {
        let g = node.grad
        let p = node.parents[0]
        let mask = tensor_mask_relu(self.nodes[p].value)
        let back = tensor_hadamard(g, mask)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
      MatMul => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        let a = self.nodes[p0].value
        let b = self.nodes[p1].value
        let shape_a = a.get_shape()
        let shape_b = b.get_shape()
        let m = shape_a[0]
        let k = shape_a[1]
        let n = shape_b[1]
        let grad_a = @math.Tensor::zeros([m, k])
        let grad_b = @math.Tensor::zeros([k, n])
        for i = 0; i < m; i = i + 1 {
          for j = 0; j < n; j = j + 1 {
            let gij = g.get([i, j])
            for p = 0; p < k; p = p + 1 {
              grad_a.set([i, p], grad_a.get([i, p]) + gij * b.get([p, j]))
              grad_b.set([p, j], grad_b.get([p, j]) + a.get([i, p]) * gij)
            }
          }
        }
        tensor_add_inplace(self.nodes[p0].grad, grad_a)
        tensor_add_inplace(self.nodes[p1].grad, grad_b)
      }
      AddBias => {
        let g = node.grad
        let p0 = node.parents[0]
        let p1 = node.parents[1]
        tensor_add_inplace(self.nodes[p0].grad, g)
        let shape_g = g.get_shape()
        let n = shape_g[0]
        let m = shape_g[1]
        let grad_b = @math.Tensor::zeros([m])
        for i = 0; i < n; i = i + 1 {
          for j = 0; j < m; j = j + 1 {
            grad_b.set([j], grad_b.get([j]) + g.get([i, j]))
          }
        }
        tensor_add_inplace(self.nodes[p1].grad, grad_b)
      }
      Sum => {
        let g = node.grad
        let p = node.parents[0]
        let scalar = g.get([0])
        let base = tensor_ones_like(self.nodes[p].value)
        let back = tensor_scale(base, scalar)
        tensor_add_inplace(self.nodes[p].grad, back)
      }
    }
  }
}

///|
pub fn ComputationGraph::grad_of(
  self : ComputationGraph,
  id : Int,
) -> @math.Tensor {
  self.nodes[id].grad
}

///|
pub fn ComputationGraph::value_of(
  self : ComputationGraph,
  id : Int,
) -> @math.Tensor {
  self.nodes[id].value
}
