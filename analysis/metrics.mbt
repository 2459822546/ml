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

///|
/// 二元对数损失（log loss）
pub fn log_loss(
  y_true : @math.Vector,
  y_proba : @math.Vector,
  eps? : Double = 1.0e-15,
) -> Double {
  let n = ensure_same_size(y_true, y_proba)
  let mut loss = 0.0
  for i = 0; i < n; i = i + 1 {
    let y = y_true.get(i)
    let mut p = y_proba.get(i)
    if not(y == 0.0 || y == 1.0) {
      abort("仅支持 {0,1} 标签")
    }
    if p < eps {
      p = eps
    }
    if p > 1.0 - eps {
      p = 1.0 - eps
    }
    loss = loss -
      y * @moonbitlang/core/math.ln(p) -
      (1.0 - y) * @moonbitlang/core/math.ln(1.0 - p)
  }
  loss / n.to_double()
}

///|
fn sorted_indices_desc(values : @math.Vector) -> Array[Int] {
  let n = values.size()
  let idx = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    idx[i] = i
  }
  idx.sort_by(fn(a, b) {
    let va = values.get(a)
    let vb = values.get(b)
    if va > vb {
      -1
    } else if va < vb {
      1
    } else {
      0
    }
  })
  idx
}

///|
/// ROC AUC（二分类，基于分数降序梯形积分）
pub fn roc_auc_score(y_true : @math.Vector, y_score : @math.Vector) -> Double {
  let n = ensure_same_size(y_true, y_score)
  let mut pos = 0
  let mut neg = 0
  for i = 0; i < n; i = i + 1 {
    let y = y_true.get(i)
    if y == 1.0 {
      pos = pos + 1
    } else if y == 0.0 {
      neg = neg + 1
    } else {
      abort("仅支持 {0,1} 标签")
    }
  }
  if pos == 0 || neg == 0 {
    return 0.5
  }
  let idx = sorted_indices_desc(y_score)
  let mut tp = 0.0
  let mut fp = 0.0
  let mut prev_score = @double.infinity
  let mut area = 0.0
  let mut fpr_prev = 0.0
  let mut tpr_prev = 0.0
  for k = 0; k < idx.length(); k = k + 1 {
    let i = idx[k]
    let s = y_score.get(i)
    if s != prev_score {
      let fpr = fp / neg.to_double()
      let tpr = tp / pos.to_double()
      area = area + (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
      fpr_prev = fpr
      tpr_prev = tpr
      prev_score = s
    }
    if y_true.get(i) == 1.0 {
      tp = tp + 1.0
    } else {
      fp = fp + 1.0
    }
  }
  let fpr = fp / neg.to_double()
  let tpr = tp / pos.to_double()
  area = area + (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
  area
}

///|
/// PR AUC（二分类，平均精确率积累）
pub fn pr_auc_score(y_true : @math.Vector, y_score : @math.Vector) -> Double {
  let n = ensure_same_size(y_true, y_score)
  let mut pos = 0
  for i = 0; i < n; i = i + 1 {
    let y = y_true.get(i)
    if y == 1.0 {
      pos = pos + 1
    } else if not(y == 0.0) {
      abort("仅支持 {0,1} 标签")
    }
  }
  if pos == 0 {
    return 0.0
  }
  let idx = sorted_indices_desc(y_score)
  let mut tp = 0.0
  let mut fp = 0.0
  let mut prev_recall = 0.0
  let mut area = 0.0
  for k = 0; k < idx.length(); k = k + 1 {
    let i = idx[k]
    if y_true.get(i) == 1.0 {
      tp = tp + 1.0
    } else {
      fp = fp + 1.0
    }
    let recall = tp / pos.to_double()
    let precision = tp / (tp + fp)
    area = area + (recall - prev_recall) * precision
    prev_recall = recall
  }
  area
}

///|
fn unique_sorted_labels(labels : @math.Vector) -> Array[Double] {
  let uniq = []
  for i = 0; i < labels.size(); i = i + 1 {
    let v = labels.get(i)
    let mut found = false
    for j = 0; j < uniq.length(); j = j + 1 {
      if uniq[j] == v {
        found = true
        break
      }
    }
    if not(found) {
      uniq.push(v)
    }
  }
  uniq.sort_by(fn(a, b) { if a < b { -1 } else if a > b { 1 } else { 0 } })
  uniq
}

///|
/// 混淆矩阵（按升序类别排列）
pub fn confusion_matrix(
  y_true : @math.Vector,
  y_pred : @math.Vector,
) -> Array[Array[Int]] {
  let n = ensure_same_size(y_true, y_pred)
  let classes = unique_sorted_labels(y_true)
  let matrix = Array::make(classes.length(), [])
  for i = 0; i < classes.length(); i = i + 1 {
    matrix[i] = Array::make(classes.length(), 0)
  }
  for i = 0; i < n; i = i + 1 {
    let t = y_true.get(i)
    let p = y_pred.get(i)
    let mut t_idx = -1
    let mut p_idx = -1
    for j = 0; j < classes.length(); j = j + 1 {
      if classes[j] == t {
        t_idx = j
      }
      if classes[j] == p {
        p_idx = j
      }
    }
    if t_idx == -1 || p_idx == -1 {
      abort("预测或真实标签未在类别集合中")
    }
    matrix[t_idx][p_idx] = matrix[t_idx][p_idx] + 1
  }
  matrix
}

///|
/// 按类别精确率与召回率（与混淆矩阵同顺序）
pub fn per_class_precision_recall(
  y_true : @math.Vector,
  y_pred : @math.Vector,
) -> (Array[Double], Array[Double]) {
  let cm = confusion_matrix(y_true, y_pred)
  let n_classes = cm.length()
  let precision = Array::make(n_classes, 0.0)
  let recall = Array::make(n_classes, 0.0)
  for i = 0; i < n_classes; i = i + 1 {
    let tp = cm[i][i].to_double()
    let mut col_sum = 0.0
    let mut row_sum = 0.0
    for j = 0; j < n_classes; j = j + 1 {
      col_sum = col_sum + cm[j][i].to_double()
      row_sum = row_sum + cm[i][j].to_double()
    }
    if col_sum > 0.0 {
      precision[i] = tp / col_sum
    }
    if row_sum > 0.0 {
      recall[i] = tp / row_sum
    }
  }
  (precision, recall)
}

///|
/// 中位数绝对误差
pub fn median_absolute_error(
  y_true : @math.Vector,
  y_pred : @math.Vector,
) -> Double {
  let n = ensure_same_size(y_true, y_pred)
  let residuals = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    residuals[i] = (y_true.get(i) - y_pred.get(i)).abs()
  }
  @math.median(residuals)
}

///|
/// 调整后的 R²，需提供特征数
pub fn adjusted_r2_score(
  y_true : @math.Vector,
  y_pred : @math.Vector,
  n_features : Int,
) -> Double {
  let n = ensure_same_size(y_true, y_pred)
  if n_features < 0 {
    abort("n_features 不能为负")
  }
  if n <= n_features + 1 {
    return r2_score(y_true, y_pred)
  }
  let r2 = r2_score(y_true, y_pred)
  1.0 -
  (1.0 - r2) *
  (n.to_double() - 1.0) /
  (n.to_double() - n_features.to_double() - 1.0)
}

///|
fn pairwise_distance(a : Array[Double], b : Array[Double]) -> Double {
  if a.length() != b.length() {
    abort("向量长度不一致")
  }
  let mut sum = 0.0
  for i = 0; i < a.length(); i = i + 1 {
    let diff = a[i] - b[i]
    sum = sum + diff * diff
  }
  sum.sqrt()
}

///|
fn cluster_indices(labels : Array[Int]) -> Array[Int] {
  let uniq = []
  for i = 0; i < labels.length(); i = i + 1 {
    let v = labels[i]
    let mut found = false
    for j = 0; j < uniq.length(); j = j + 1 {
      if uniq[j] == v {
        found = true
        break
      }
    }
    if not(found) {
      uniq.push(v)
    }
  }
  uniq
}

///|
/// Silhouette 系数（欧氏距离）
pub fn silhouette_score(
  data : Array[Array[Double]],
  labels : Array[Int],
) -> Double {
  if data.length() == 0 || data.length() != labels.length() {
    abort("数据与标签长度不一致或为空")
  }
  let n_samples = data.length()
  let uniq = cluster_indices(labels)
  if uniq.length() < 2 {
    abort("至少需要两个簇")
  }
  let mut score_sum = 0.0
  for i = 0; i < n_samples; i = i + 1 {
    let label_i = labels[i]
    let mut a = 0.0
    let mut count_in = 0
    for j = 0; j < n_samples; j = j + 1 {
      if labels[j] == label_i && i != j {
        a = a + pairwise_distance(data[i], data[j])
        count_in = count_in + 1
      }
    }
    if count_in > 0 {
      a = a / count_in.to_double()
    }
    let mut b = @double.infinity
    for c_idx = 0; c_idx < uniq.length(); c_idx = c_idx + 1 {
      let label_c = uniq[c_idx]
      if label_c == label_i {
        continue
      }
      let mut dist_sum = 0.0
      let mut count_out = 0
      for j = 0; j < n_samples; j = j + 1 {
        if labels[j] == label_c {
          dist_sum = dist_sum + pairwise_distance(data[i], data[j])
          count_out = count_out + 1
        }
      }
      if count_out > 0 {
        let avg = dist_sum / count_out.to_double()
        if avg < b {
          b = avg
        }
      }
    }
    let denom = if a > b { a } else { b }
    if denom > 0.0 {
      score_sum = score_sum + (b - a) / denom
    }
  }
  score_sum / n_samples.to_double()
}

///|
/// Davies-Bouldin 指数（值越低越好）
pub fn davies_bouldin_score(
  data : Array[Array[Double]],
  labels : Array[Int],
) -> Double {
  if data.length() == 0 || data.length() != labels.length() {
    abort("数据与标签长度不一致或为空")
  }
  let uniq = cluster_indices(labels)
  let k = uniq.length()
  if k < 2 {
    abort("至少需要两个簇")
  }
  let centroids = Array::make(k, [])
  let scatter = Array::make(k, 0.0)
  for idx = 0; idx < k; idx = idx + 1 {
    let label = uniq[idx]
    let mut count = 0
    let sum = Array::make(data[0].length(), 0.0)
    for i = 0; i < data.length(); i = i + 1 {
      if labels[i] == label {
        count = count + 1
        for f = 0; f < sum.length(); f = f + 1 {
          sum[f] = sum[f] + data[i][f]
        }
      }
    }
    if count == 0 {
      abort("簇包含 0 个样本")
    }
    for f = 0; f < sum.length(); f = f + 1 {
      sum[f] = sum[f] / count.to_double()
    }
    centroids[idx] = sum
    let mut total = 0.0
    for i = 0; i < data.length(); i = i + 1 {
      if labels[i] == label {
        total = total + pairwise_distance(data[i], sum)
      }
    }
    scatter[idx] = total / count.to_double()
  }
  let mut db_sum = 0.0
  for i = 0; i < k; i = i + 1 {
    let mut worst = 0.0
    for j = 0; j < k; j = j + 1 {
      if i == j {
        continue
      }
      let dist = pairwise_distance(centroids[i], centroids[j])
      if dist == 0.0 {
        continue
      }
      let r = (scatter[i] + scatter[j]) / dist
      if r > worst {
        worst = r
      }
    }
    db_sum = db_sum + worst
  }
  db_sum / k.to_double()
}
