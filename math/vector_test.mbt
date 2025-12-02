///|
/// Vector 测试
test "Vector - 创建和基本属性" {
  let v = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_eq(v.size(), 5)
  assert_eq(v.get(0), 1.0)
  assert_eq(v.get(4), 5.0)
  let zeros = Vector::zeros(3)
  assert_eq(zeros.size(), 3)
  assert_eq(zeros.get(0), 0.0)
  let ones = Vector::ones(3)
  assert_eq(ones.get(0), 1.0)
}

///|
test "Vector - 加法和减法" {
  let v1 = Vector::new([1.0, 2.0, 3.0])
  let v2 = Vector::new([4.0, 5.0, 6.0])
  let v_add = v1.op_add(v2)
  assert_eq(v_add.get(0), 5.0)
  assert_eq(v_add.get(1), 7.0)
  assert_eq(v_add.get(2), 9.0)
  let v_sub = v2.op_sub(v1)
  assert_eq(v_sub.get(0), 3.0)
  assert_eq(v_sub.get(1), 3.0)
  assert_eq(v_sub.get(2), 3.0)
}

///|
test "Vector - 标量乘法" {
  let v = Vector::new([1.0, 2.0, 3.0])
  let v_mul = v.op_mul(2.0)
  assert_eq(v_mul.get(0), 2.0)
  assert_eq(v_mul.get(1), 4.0)
  assert_eq(v_mul.get(2), 6.0)
}

///|
test "Vector - 点积" {
  let v1 = Vector::new([1.0, 2.0, 3.0])
  let v2 = Vector::new([4.0, 5.0, 6.0])
  let dot = v1.dot(v2)
  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  assert_eq(dot, 32.0)
}

///|
test "Vector - 范数" {
  let v = Vector::new([3.0, 4.0])

  // L2 范数: sqrt(9 + 16) = 5
  let norm = v.norm_l2()
  assert_true((norm - 5.0).abs() < 0.0001)

  // L1 范数: 3 + 4 = 7
  assert_eq(v.norm_l1(), 7.0)
}

///|
test "Vector - 统计函数" {
  let v = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0])
  assert_eq(v.sum(), 15.0)
  assert_eq(v.mean(), 3.0)
  assert_eq(v.min(), 1.0)
  assert_eq(v.max(), 5.0)

  // 方差: ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = 10/5 = 2
  let vars = v.variance()
  assert_true((vars - 2.0).abs() < 0.0001)
}

///|
test "Vector - 归一化" {
  let v = Vector::new([3.0, 4.0])
  let normalized = v.normalize()

  // 归一化后的范数应该为 1
  let norm = normalized.norm_l2()
  assert_true((norm - 1.0).abs() < 0.0001)

  // 归一化后的值应该是 [0.6, 0.8]
  assert_true((normalized.get(0) - 0.6).abs() < 0.0001)
  assert_true((normalized.get(1) - 0.8).abs() < 0.0001)
}
