///|
/// 简单模型评价指标（分类/回归）

///|
/// 检查 y_true 与 y_pred 尺寸一致
fn ensure_same_size(y_true : @math.Vector, y_pred : @math.Vector) -> Int {
  if y_true.size() != y_pred.size() {
    abort("y_true 和 y_pred 长度不匹配")
  }
  y_true.size()
}

///|
/// 统计二元分类的 TP/FP/TN/FN
fn binary_counts(
  y_true : @math.Vector,
  y_pred : @math.Vector,
) -> (Int, Int, Int, Int) {
  let n = ensure_same_size(y_true, y_pred)
  let mut tp = 0
  let mut fp = 0
  let mut tn = 0
  let mut fn_count = 0
  for i = 0; i < n; i = i + 1 {
    let t = y_true.get(i)
    let p = y_pred.get(i)
    if not((t == 0.0 || t == 1.0) && (p == 0.0 || p == 1.0)) {
      abort("仅支持标签 {0,1}")
    }
    if p == 1.0 && t == 1.0 {
      tp = tp + 1
    } else if p == 1.0 && t == 0.0 {
      fp = fp + 1
    } else if p == 0.0 && t == 0.0 {
      tn = tn + 1
    } else {
      fn_count = fn_count + 1
    }
  }
  (tp, fp, tn, fn_count)
}

///|
/// 分类准确率
pub fn accuracy_score(y_true : @math.Vector, y_pred : @math.Vector) -> Double {
  let n = ensure_same_size(y_true, y_pred)
  let mut correct = 0
  for i = 0; i < n; i = i + 1 {
    if y_true.get(i) == y_pred.get(i) {
      correct = correct + 1
    }
  }
  correct.to_double() / n.to_double()
}

///|
/// 精确率 (binary)
pub fn precision_score(
  y_true : @math.Vector,
  y_pred : @math.Vector,
  eps? : Double = 1.0e-12,
) -> Double {
  let (tp, fp, _, _) = binary_counts(y_true, y_pred)
  let denom = tp.to_double() + fp.to_double()
  if denom == 0.0 {
    return eps
  }
  tp.to_double() / denom
}

///|
/// 召回率 (binary)
pub fn recall_score(
  y_true : @math.Vector,
  y_pred : @math.Vector,
  eps? : Double = 1.0e-12,
) -> Double {
  let (tp, _, _, fn_count) = binary_counts(y_true, y_pred)
  let denom = tp.to_double() + fn_count.to_double()
  if denom == 0.0 {
    return eps
  }
  tp.to_double() / denom
}

///|
/// F1 分数 (binary)
pub fn f1_score(
  y_true : @math.Vector,
  y_pred : @math.Vector,
  eps? : Double = 1.0e-12,
) -> Double {
  let p = precision_score(y_true, y_pred, eps~)
  let r = recall_score(y_true, y_pred, eps~)
  if p + r == 0.0 {
    return eps
  }
  2.0 * p * r / (p + r)
}

///|
/// MAE 平均绝对误差
pub fn mae(y_true : @math.Vector, y_pred : @math.Vector) -> Double {
  let n = ensure_same_size(y_true, y_pred)
  let mut sum = 0.0
  for i = 0; i < n; i = i + 1 {
    sum = sum + (y_true.get(i) - y_pred.get(i)).abs()
  }
  sum / n.to_double()
}

///|
/// MSE 均方误差
pub fn mse(y_true : @math.Vector, y_pred : @math.Vector) -> Double {
  let n = ensure_same_size(y_true, y_pred)
  let mut sum = 0.0
  for i = 0; i < n; i = i + 1 {
    let diff = y_true.get(i) - y_pred.get(i)
    sum = sum + diff * diff
  }
  sum / n.to_double()
}

///|
/// RMSE 均方根误差
pub fn rmse(y_true : @math.Vector, y_pred : @math.Vector) -> Double {
  mse(y_true, y_pred).sqrt()
}

///|
/// R² 判定系数
pub fn r2_score(y_true : @math.Vector, y_pred : @math.Vector) -> Double {
  let n = ensure_same_size(y_true, y_pred)
  let y_mean = y_true.mean()
  let mut ss_tot = 0.0
  let mut ss_res = 0.0
  for i = 0; i < n; i = i + 1 {
    let diff = y_true.get(i) - y_mean
    ss_tot = ss_tot + diff * diff
    let residual = y_true.get(i) - y_pred.get(i)
    ss_res = ss_res + residual * residual
  }
  1.0 - ss_res / ss_tot
}
