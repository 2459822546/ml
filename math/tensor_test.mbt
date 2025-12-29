///|
/// 张量测试
test "Tensor - 创建和基本属性" {
  // 创建3D张量 (2x3x4)
  let data = Array::make(24, 0.0)
  for i = 0; i < 24; i = i + 1 {
    data[i] = i.to_double()
  }
  let tensor = Tensor::new(data, [2, 3, 4])
  assert_eq(tensor.ndim(), 3)
  assert_eq(tensor.total_size(), 24)
  let shape = tensor.get_shape()
  assert_eq(shape[0], 2)
  assert_eq(shape[1], 3)
  assert_eq(shape[2], 4)
}

///|
test "Tensor - zeros/ones/fill" {
  let zeros_t = Tensor::zeros([2, 3])
  assert_eq(zeros_t.get([0, 0]), 0.0)
  assert_eq(zeros_t.get([1, 2]), 0.0)
  let ones_t = Tensor::ones([2, 3])
  assert_eq(ones_t.get([0, 0]), 1.0)
  assert_eq(ones_t.get([1, 2]), 1.0)
  let fill_t = Tensor::fill([2, 3], 5.0)
  assert_eq(fill_t.get([0, 0]), 5.0)
  assert_eq(fill_t.get([1, 2]), 5.0)
}

///|
test "Tensor - 从Vector和Matrix创建" {
  // 从Vector创建
  let vec = Vector::new([1.0, 2.0, 3.0])
  let t1 = Tensor::from_vector(vec)
  assert_eq(t1.ndim(), 1)
  assert_eq(t1.get([0]), 1.0)
  assert_eq(t1.get([2]), 3.0)

  // 从Matrix创建
  let mat = Matrix::new([[1.0, 2.0], [3.0, 4.0]])
  let t2 = Tensor::from_matrix(mat)
  assert_eq(t2.ndim(), 2)
  assert_eq(t2.get([0, 0]), 1.0)
  assert_eq(t2.get([1, 1]), 4.0)
}

///|
test "Tensor - get/set" {
  let tensor = Tensor::zeros([2, 3, 2])
  tensor.set([0, 0, 0], 1.0)
  tensor.set([1, 2, 1], 5.0)
  assert_eq(tensor.get([0, 0, 0]), 1.0)
  assert_eq(tensor.get([1, 2, 1]), 5.0)
  assert_eq(tensor.get([0, 1, 0]), 0.0)
}

///|
test "Tensor - 加法" {
  let t1 = Tensor::fill([2, 3], 2.0)
  let t2 = Tensor::fill([2, 3], 3.0)
  let result = t1 + t2
  assert_eq(result.get([0, 0]), 5.0)
  assert_eq(result.get([1, 2]), 5.0)
}

///|
test "Tensor - 减法" {
  let t1 = Tensor::fill([2, 3], 5.0)
  let t2 = Tensor::fill([2, 3], 2.0)
  let result = t1 - t2
  assert_eq(result.get([0, 0]), 3.0)
  assert_eq(result.get([1, 2]), 3.0)
}

///|
test "Tensor - 标量乘法" {
  let t1 = Tensor::fill([2, 3], 2.0)
  let result = t1.scalar_mul(3.0)
  assert_eq(result.get([0, 0]), 6.0)
  assert_eq(result.get([1, 2]), 6.0)
}

///|
test "Tensor - Hadamard积 (元素级乘法)" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2])
  let t2 = Tensor::new([2.0, 2.0, 2.0, 2.0], [2, 2])
  let result = t1 * t2 // 使用 * 运算符
  assert_eq(result.get([0, 0]), 2.0)
  assert_eq(result.get([0, 1]), 4.0)
  assert_eq(result.get([1, 0]), 6.0)
  assert_eq(result.get([1, 1]), 8.0)
}

///|
test "Tensor - reshape" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let t2 = t1.reshape([3, 2])
  assert_eq(t2.ndim(), 2)
  let shape = t2.get_shape()
  assert_eq(shape[0], 3)
  assert_eq(shape[1], 2)
  assert_eq(t2.get([0, 0]), 1.0)
  assert_eq(t2.get([2, 1]), 6.0)
}

///|
test "Tensor - transpose (2D)" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let t2 = t1.transpose()
  let shape = t2.get_shape()
  assert_eq(shape[0], 3)
  assert_eq(shape[1], 2)
  assert_eq(t2.get([0, 0]), 1.0)
  assert_eq(t2.get([1, 0]), 2.0)
  assert_eq(t2.get([0, 1]), 4.0)
}

///|
test "Tensor - transpose (3D axes)" {
  let t1 = Tensor::new(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    [2, 3, 2],
  )
  let t2 = t1.transpose(axes=[1, 0, 2])
  let shape = t2.get_shape()
  assert_eq(shape[0], 3)
  assert_eq(shape[1], 2)
  assert_eq(shape[2], 2)
  assert_eq(t2.get([0, 0, 0]), 0.0)
  assert_eq(t2.get([1, 0, 1]), 3.0)
  assert_eq(t2.get([2, 1, 1]), 11.0)
}

///|
test "Tensor - flatten" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2])
  let t2 = t1.flatten()
  assert_eq(t2.ndim(), 1)
  assert_eq(t2.get([0]), 1.0)
  assert_eq(t2.get([3]), 4.0)
}

///|
test "Tensor - slice" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let t2 = t1.slice(1, 1, end=3)
  let shape = t2.get_shape()
  assert_eq(shape[0], 2)
  assert_eq(shape[1], 2)
  assert_eq(t2.get([0, 0]), 2.0)
  assert_eq(t2.get([0, 1]), 3.0)
  assert_eq(t2.get([1, 0]), 5.0)
  assert_eq(t2.get([1, 1]), 6.0)
}

///|
test "Tensor - concat" {
  let a = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2])
  let b = Tensor::new([5.0, 6.0, 7.0, 8.0], [2, 2])
  let c = Tensor::concat(0, [a, b])
  let shape = c.get_shape()
  assert_eq(shape[0], 4)
  assert_eq(shape[1], 2)
  assert_eq(c.get([2, 0]), 5.0)
  assert_eq(c.get([3, 1]), 8.0)
}

///|
test "Tensor - stack" {
  let a = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2])
  let b = Tensor::new([5.0, 6.0, 7.0, 8.0], [2, 2])
  let c = Tensor::stack(0, [a, b])
  let shape = c.get_shape()
  assert_eq(shape[0], 2)
  assert_eq(shape[1], 2)
  assert_eq(shape[2], 2)
  assert_eq(c.get([0, 0, 0]), 1.0)
  assert_eq(c.get([0, 1, 1]), 4.0)
  assert_eq(c.get([1, 0, 0]), 5.0)
  assert_eq(c.get([1, 1, 1]), 8.0)
}

///|
test "Tensor - broadcast add" {
  let a = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let b = Tensor::new([10.0, 20.0, 30.0], [3])
  let c = a + b
  let shape = c.get_shape()
  assert_eq(shape[0], 2)
  assert_eq(shape[1], 3)
  assert_eq(c.get([0, 1]), 22.0)
  assert_eq(c.get([1, 2]), 36.0)
}

///|
test "Tensor - broadcast mul" {
  let a = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 1, 2])
  let b = Tensor::new([10.0, 20.0, 30.0, 40.0], [2, 2])
  let c = a * b
  let shape = c.get_shape()
  assert_eq(shape[0], 2)
  assert_eq(shape[1], 2)
  assert_eq(shape[2], 2)
  assert_eq(c.get([0, 0, 0]), 10.0)
  assert_eq(c.get([1, 1, 1]), 160.0)
}

///|
test "Tensor - broadcast div" {
  let a = Tensor::new([2.0, 4.0, 6.0, 8.0], [2, 2])
  let b = Tensor::new([1.0, 2.0], [2, 1])
  let c = a / b
  let shape = c.get_shape()
  assert_eq(shape[0], 2)
  assert_eq(shape[1], 2)
  assert_eq(c.get([0, 0]), 2.0)
  assert_eq(c.get([0, 1]), 4.0)
  assert_eq(c.get([1, 0]), 3.0)
  assert_eq(c.get([1, 1]), 4.0)
}

///|
test "Tensor - sum_axis" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

  // 沿axis=0求和 (按行求和)
  let sum_axis0 = t1.sum_axis(0)
  assert_eq(sum_axis0.get([0]), 5.0) // 1+4
  assert_eq(sum_axis0.get([1]), 7.0) // 2+5
  assert_eq(sum_axis0.get([2]), 9.0) // 3+6

  // 沿axis=1求和 (按列求和)
  let sum_axis1 = t1.sum_axis(1)
  assert_eq(sum_axis1.get([0]), 6.0) // 1+2+3
  assert_eq(sum_axis1.get([1]), 15.0) // 4+5+6
}

///|
test "Tensor - sum_axis (3D)" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  let sum_axis0 = t1.sum_axis(0)
  assert_eq(sum_axis0.get([0, 0]), 6.0)
  assert_eq(sum_axis0.get([0, 1]), 8.0)
  assert_eq(sum_axis0.get([1, 0]), 10.0)
  assert_eq(sum_axis0.get([1, 1]), 12.0)
}

///|
test "Tensor - sum_axes" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  let sum_axes = t1.sum_axes([0, 2])
  let shape = sum_axes.get_shape()
  assert_eq(shape[0], 2)
  assert_eq(sum_axes.get([0]), 14.0)
  assert_eq(sum_axes.get([1]), 22.0)
}

///|
test "Tensor - sum_axes keepdim" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  let sum_axes = t1.sum_axes([0, 2], keepdim=true)
  let shape = sum_axes.get_shape()
  assert_eq(shape[0], 1)
  assert_eq(shape[1], 2)
  assert_eq(shape[2], 1)
  assert_eq(sum_axes.get([0, 0, 0]), 14.0)
  assert_eq(sum_axes.get([0, 1, 0]), 22.0)
}

///|
test "Tensor - mean_axis" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  let mean_axis2 = t1.mean_axis(2)
  assert_true((mean_axis2.get([0, 0]) - 1.5).abs() < 1.0e-9)
  assert_true((mean_axis2.get([0, 1]) - 3.5).abs() < 1.0e-9)
  assert_true((mean_axis2.get([1, 0]) - 5.5).abs() < 1.0e-9)
  assert_true((mean_axis2.get([1, 1]) - 7.5).abs() < 1.0e-9)
}

///|
test "Tensor - mean_axes" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  let mean_axes = t1.mean_axes([0, 2])
  let shape = mean_axes.get_shape()
  assert_eq(shape[0], 2)
  assert_true((mean_axes.get([0]) - 3.5).abs() < 1.0e-9)
  assert_true((mean_axes.get([1]) - 5.5).abs() < 1.0e-9)
}

///|
test "Tensor - 统计函数" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  assert_eq(t1.sum(), 21.0)
  assert_eq(t1.mean(), 3.5)
  assert_eq(t1.min(), 1.0)
  assert_eq(t1.max(), 6.0)
}

///|
test "Tensor - copy" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2])
  let t2 = t1.copy()
  t2.set([0, 0], 99.0)
  assert_eq(t1.get([0, 0]), 1.0) // 原张量不变
  assert_eq(t2.get([0, 0]), 99.0)
}

///|
test "Tensor - map" {
  let t1 = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2])
  let t2 = t1.map(fn(x) { x * 2.0 })
  assert_eq(t2.get([0, 0]), 2.0)
  assert_eq(t2.get([0, 1]), 4.0)
  assert_eq(t2.get([1, 0]), 6.0)
  assert_eq(t2.get([1, 1]), 8.0)
}

///|
test "Tensor - 转换为Vector和Matrix" {
  // 1D张量转Vector
  let t1 = Tensor::new([1.0, 2.0, 3.0], [3])
  let vec = t1.to_vector()
  assert_eq(vec.get(0), 1.0)
  assert_eq(vec.get(2), 3.0)

  // 2D张量转Matrix
  let t2 = Tensor::new([1.0, 2.0, 3.0, 4.0], [2, 2])
  let mat = t2.to_matrix()
  assert_eq(mat.get(0, 0), 1.0)
  assert_eq(mat.get(1, 1), 4.0)
}
