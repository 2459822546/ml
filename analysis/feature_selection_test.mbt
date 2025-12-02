///|
/// 特征选择工具测试

///|
test "variance_threshold drops low-variance columns" {
  let x = @math.Matrix::new([[0.0, 1.0, 0.5], [1.0, 1.0, 0.2], [2.0, 1.0, 0.7]])
  let kept = variance_threshold(x, 1.0e-6)
  assert_eq(kept.length(), 2)
  assert_eq(kept[0], 0)
  assert_eq(kept[1], 2)
}

///|
test "correlation_filter keeps weakly correlated columns" {
  let x = @math.Matrix::new([
    [1.0, 2.0, 0.0],
    [2.0, 4.0, 1.0],
    [3.0, 6.0, 0.0],
    [4.0, 8.0, 2.0],
  ])
  let kept = correlation_filter(x, 0.9)
  assert_eq(kept.length(), 2)
  // 第二列与第一列高度相关，应被过滤
  let has0 = kept[0] == 0 || kept[1] == 0
  let has2 = kept[0] == 2 || kept[1] == 2
  assert_true(has0)
  assert_true(has2)
}

///|
test "select_by_mutual_information prefers informative column" {
  let x = @math.Matrix::new([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
  let y = @math.Vector::new([0.0, 0.0, 1.0, 1.0])
  let selected = select_by_mutual_information(x, y, 1, n_bins=5)
  assert_eq(selected.length(), 1)
  assert_eq(selected[0], 0)
}

///|
test "rfe_linear_regression drops smallest weight feature" {
  let x = @math.Matrix::new([
    [1.0, 2.0, 0.0],
    [2.0, 1.0, 0.0],
    [3.0, 1.0, 0.0],
    [4.0, 1.0, 0.0],
  ])
  let y = @math.Vector::new([3.0, 6.0, 9.0, 12.0])
  let kept = rfe_linear_regression(x, y, 2)
  assert_eq(kept.length(), 2)
  let has0 = kept[0] == 0 || kept[1] == 0
  let has1 = kept[0] == 1 || kept[1] == 1
  assert_true(has0)
  assert_true(has1)
}
