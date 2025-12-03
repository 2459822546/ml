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
pub enum Op {
  Input
  Add
  Sub
  Mul
  Relu
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
