///|
/// GaussianMixture 简单测试

///|
test "gmm two clusters" {
  let data = @math.Matrix::new([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 3.9]])
  let gmm = GaussianMixture::new(2, max_iter=20, tol=1.0e-3)
  gmm.fit(data)
  let resp = gmm.predict_proba(data)
  let (_, k) = resp.shape()
  assert_eq(k, 2)
  for i = 0; i < 4; i = i + 1 {
    let row = resp.get_row(i)
    let sum = row.sum()
    assert_true((sum - 1.0).abs() < 1.0e-6)
  }
  let labels = gmm.predict(data)
  assert_eq(labels.length(), 4)
}
