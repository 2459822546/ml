///|
/// 假设检验与分布统计测试

///|
test "t_test_independent produces expected sign" {
  let a = [1.0, 2.0, 3.0, 4.0, 5.0]
  let b = [2.0, 3.0, 4.0, 5.0, 6.0]
  let t_stat = t_test_independent(a, b)
  assert_true(t_stat < 0.0) // a 均值更小
}

///|
test "chi_square_gof returns zero when observed equals expected" {
  let observed = [10.0, 15.0, 5.0]
  let expected = [10.0, 15.0, 5.0]
  let chi2 = chi_square_gof(observed, expected)
  assert_eq(chi2, 0.0)
}

///|
test "ks_test_two_sample detects separated samples" {
  let a = [0.0, 0.0, 0.0, 0.0]
  let b = [1.0, 1.0, 1.0, 1.0]
  let d = ks_test_two_sample(a, b)
  assert_eq(d, 1.0)
}

///|
test "jarque_bera is small for near-normal data and larger for skewed data" {
  let normalish = [-1.0, -0.5, 0.0, 0.5, 1.0]
  let skewed = [0.1, 0.2, 0.3, 5.0, 6.0]
  let jb_normal = jarque_bera(normalish)
  let jb_skewed = jarque_bera(skewed)
  assert_true(jb_normal < jb_skewed)
}
