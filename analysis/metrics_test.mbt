///|
/// 简单模型评价指标测试

///|
test "accuracy precision recall f1" {
  let y_true = @math.Vector::new([1.0, 0.0, 1.0, 1.0, 0.0])
  let y_pred = @math.Vector::new([1.0, 0.0, 0.0, 1.0, 1.0])
  let acc = accuracy_score(y_true, y_pred)
  let prec = precision_score(y_true, y_pred)
  let rec = recall_score(y_true, y_pred)
  let f1 = f1_score(y_true, y_pred)
  assert_true((acc - 0.6).abs() < 1.0e-9)
  assert_true((prec - 0.6666666667).abs() < 1.0e-6)
  assert_true((rec - 0.6666666667).abs() < 1.0e-6)
  assert_true((f1 - 0.6666666667).abs() < 1.0e-6)
}

///|
test "regression metrics" {
  let y_true = @math.Vector::new([1.0, 2.0, 3.0])
  let y_pred = @math.Vector::new([1.5, 1.5, 3.0])
  let mae_val = mae(y_true, y_pred)
  let mse_val = mse(y_true, y_pred)
  let rmse_val = rmse(y_true, y_pred)
  let r2_val = r2_score(y_true, y_pred)
  assert_true((mae_val - 0.3333333333).abs() < 1.0e-6)
  assert_true((mse_val - 0.1666666667).abs() < 1.0e-6)
  assert_true((rmse_val - 0.4082482904).abs() < 1.0e-6)
  assert_true(r2_val > 0.5)
}
