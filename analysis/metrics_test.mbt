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

///|
test "binary probabilistic metrics" {
  let y_true = @math.Vector::new([1.0, 0.0, 1.0, 0.0])
  let y_score = @math.Vector::new([0.9, 0.2, 0.8, 0.1])
  let logloss = log_loss(y_true, y_score)
  let auc = roc_auc_score(y_true, y_score)
  let pr_auc = pr_auc_score(y_true, y_score)
  assert_true(logloss < 0.5)
  assert_true((auc - 1.0).abs() < 1.0e-9)
  assert_true(pr_auc > 0.8)
}

///|
test "confusion matrix per-class" {
  let y_true = @math.Vector::new([0.0, 1.0, 2.0, 2.0])
  let y_pred = @math.Vector::new([0.0, 2.0, 2.0, 1.0])
  let cm = confusion_matrix(y_true, y_pred)
  assert_eq(cm.length(), 3)
  assert_eq(cm[0][0], 1)
  assert_eq(cm[1][2], 1)
  assert_eq(cm[2][1], 1)
  let (prec, rec) = per_class_precision_recall(y_true, y_pred)
  // class 2 precision: tp=1, pred2 total=2 -> 0.5
  assert_true((prec[2] - 0.5).abs() < 1.0e-9)
  // class 2 recall: tp=1, actual2 total=2 -> 0.5
  assert_true((rec[2] - 0.5).abs() < 1.0e-9)
}

///|
test "regression extras" {
  let y_true = @math.Vector::new([1.0, 2.0, 3.0, 4.0])
  let y_pred = @math.Vector::new([1.0, 2.5, 2.5, 4.5])
  let med = median_absolute_error(y_true, y_pred)
  let adj_r2 = adjusted_r2_score(y_true, y_pred, 1)
  assert_true((med - 0.5).abs() < 1.0e-9)
  assert_true(adj_r2 < r2_score(y_true, y_pred))
}

///|
test "clustering metrics" {
  let data = [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]
  let labels = [0, 0, 1, 1]
  let sil = silhouette_score(data, labels)
  let dbi = davies_bouldin_score(data, labels)
  assert_true(sil > 0.7)
  assert_true(dbi < 0.6)
}
