///|
/// 简单损失与正则项（前向）

///|
pub fn mse_loss(y_true : Array[Double], y_pred : Array[Double]) -> Double {
  if y_true.length() != y_pred.length() {
    abort("长度不一致")
  }
  let mut sum = 0.0
  for i = 0; i < y_true.length(); i = i + 1 {
    let diff = y_true[i] - y_pred[i]
    sum = sum + diff * diff
  }
  sum / y_true.length().to_double()
}

///|
pub fn huber_loss(
  y_true : Array[Double],
  y_pred : Array[Double],
  delta? : Double = 1.0,
) -> Double {
  if y_true.length() != y_pred.length() {
    abort("长度不一致")
  }
  let mut sum = 0.0
  for i = 0; i < y_true.length(); i = i + 1 {
    let diff = y_true[i] - y_pred[i]
    let ad = diff.abs()
    if ad <= delta {
      sum = sum + 0.5 * diff * diff
    } else {
      sum = sum + delta * (ad - 0.5 * delta)
    }
  }
  sum / y_true.length().to_double()
}

///|
/// MAE
pub fn mae_loss(y_true : Array[Double], y_pred : Array[Double]) -> Double {
  if y_true.length() != y_pred.length() {
    abort("长度不一致")
  }
  let mut sum = 0.0
  for i = 0; i < y_true.length(); i = i + 1 {
    sum = sum + (y_true[i] - y_pred[i]).abs()
  }
  sum / y_true.length().to_double()
}

///|
/// Smooth L1
pub fn smooth_l1_loss(
  y_true : Array[Double],
  y_pred : Array[Double],
  beta? : Double = 1.0,
) -> Double {
  if y_true.length() != y_pred.length() {
    abort("长度不一致")
  }
  if beta <= 0.0 {
    abort("beta 必须大于 0")
  }
  let mut sum = 0.0
  for i = 0; i < y_true.length(); i = i + 1 {
    let diff = (y_true[i] - y_pred[i]).abs()
    if diff < beta {
      sum = sum + 0.5 * diff * diff / beta
    } else {
      sum = sum + diff - 0.5 * beta
    }
  }
  sum / y_true.length().to_double()
}

///|
/// 二分类交叉熵
pub fn binary_cross_entropy(
  y_true : Array[Double],
  y_prob : Array[Double],
  eps? : Double = 1.0e-12,
) -> Double {
  if y_true.length() != y_prob.length() {
    abort("长度不一致")
  }
  let mut loss = 0.0
  for i = 0; i < y_true.length(); i = i + 1 {
    let y = y_true[i]
    let mut p = y_prob[i]
    if not(y == 0.0 || y == 1.0) {
      abort("标签需为 {0,1}")
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
  loss / y_true.length().to_double()
}

///|
/// 多标签二分类交叉熵
pub fn multilabel_binary_cross_entropy(
  y_true : Array[Array[Double]],
  y_prob : Array[Array[Double]],
  eps? : Double = 1.0e-12,
) -> Double {
  if y_true.length() != y_prob.length() {
    abort("长度不一致")
  }
  if y_true.length() == 0 {
    return 0.0
  }
  let n = y_true.length()
  let k = y_true[0].length()
  let mut loss = 0.0
  for i = 0; i < n; i = i + 1 {
    if y_true[i].length() != k || y_prob[i].length() != k {
      abort("标签维度不一致")
    }
    for j = 0; j < k; j = j + 1 {
      let y = y_true[i][j]
      let mut p = y_prob[i][j]
      if not(y == 0.0 || y == 1.0) {
        abort("标签需为 {0,1}")
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
  }
  loss / (n * k).to_double()
}

///|
/// 二分类 Focal Loss
pub fn focal_loss_binary(
  y_true : Array[Double],
  y_prob : Array[Double],
  gamma? : Double = 2.0,
  alpha? : Double = 0.25,
  eps? : Double = 1.0e-12,
) -> Double {
  if y_true.length() != y_prob.length() {
    abort("长度不一致")
  }
  if gamma < 0.0 {
    abort("gamma 不能为负")
  }
  if alpha < 0.0 || alpha > 1.0 {
    abort("alpha 必须在 [0,1]")
  }
  let mut loss = 0.0
  for i = 0; i < y_true.length(); i = i + 1 {
    let y = y_true[i]
    let mut p = y_prob[i]
    if not(y == 0.0 || y == 1.0) {
      abort("标签需为 {0,1}")
    }
    if p < eps {
      p = eps
    }
    if p > 1.0 - eps {
      p = 1.0 - eps
    }
    let pt = if y == 1.0 { p } else { 1.0 - p }
    let w = if y == 1.0 { alpha } else { 1.0 - alpha }
    loss = loss -
      w *
      @moonbitlang/core/math.pow(1.0 - pt, gamma) *
      @moonbitlang/core/math.ln(pt)
  }
  loss / y_true.length().to_double()
}

///|
/// 多分类交叉熵，y_true 为类别索引
pub fn cross_entropy(y_true : Array[Int], logits : @math.Matrix) -> Double {
  let (n, c) = logits.shape()
  if y_true.length() != n {
    abort("长度不一致")
  }
  let mut loss = 0.0
  for i = 0; i < n; i = i + 1 {
    let cls = y_true[i]
    if cls < 0 || cls >= c {
      abort("类别索引越界")
    }
    let row = logits.get_row(i)
    let mut max_v = row.get(0)
    for j = 1; j < c; j = j + 1 {
      let v = row.get(j)
      if v > max_v {
        max_v = v
      }
    }
    let mut sum = 0.0
    let mut logprob = 0.0
    for j = 0; j < c; j = j + 1 {
      let v = @moonbitlang/core/math.exp(row.get(j) - max_v)
      sum = sum + v
      if j == cls {
        logprob = row.get(j) - max_v
      }
    }
    logprob = logprob - @moonbitlang/core/math.ln(sum)
    if logprob < -@double.infinity {
      logprob = -@double.infinity
    }
    if logprob < -1.0e9 {
      logprob = -1.0e9
    }
    if logprob > 1.0e9 {
      logprob = 1.0e9
    }
    loss = loss - logprob
  }
  loss / n.to_double()
}

///|
/// 带标签平滑的多分类交叉熵
pub fn label_smoothing_cross_entropy(
  y_true : Array[Int],
  logits : @math.Matrix,
  smoothing? : Double = 0.1,
) -> Double {
  let (n, c) = logits.shape()
  if y_true.length() != n {
    abort("长度不一致")
  }
  if smoothing < 0.0 || smoothing >= 1.0 {
    abort("smoothing 必须在 [0,1)")
  }
  let on_value = 1.0 - smoothing
  let off_value = smoothing / c.to_double()
  let mut loss = 0.0
  for i = 0; i < n; i = i + 1 {
    let cls = y_true[i]
    if cls < 0 || cls >= c {
      abort("类别索引越界")
    }
    let row = logits.get_row(i)
    let mut max_v = row.get(0)
    for j = 1; j < c; j = j + 1 {
      let v = row.get(j)
      if v > max_v {
        max_v = v
      }
    }
    let mut sum = 0.0
    let probs = Array::make(c, 0.0)
    for j = 0; j < c; j = j + 1 {
      let v = @moonbitlang/core/math.exp(row.get(j) - max_v)
      sum = sum + v
      probs[j] = v
    }
    for j = 0; j < c; j = j + 1 {
      let p = probs[j] / sum
      let t = if j == cls { on_value } else { off_value }
      let logp = @moonbitlang/core/math.ln(p)
      loss = loss - t * logp
    }
  }
  loss / n.to_double()
}

///|
/// KL 散度（p || q）
pub fn kl_divergence(p : Array[Double], q : Array[Double]) -> Double {
  if p.length() != q.length() {
    abort("长度不一致")
  }
  let mut sum = 0.0
  for i = 0; i < p.length(); i = i + 1 {
    let pi = p[i]
    let qi = q[i]
    if pi < 0.0 || qi <= 0.0 {
      abort("概率必须为正")
    }
    if pi == 0.0 {
      continue
    }
    sum = sum + pi * @moonbitlang/core/math.ln(pi / qi)
  }
  sum
}

///|
/// Hinge loss（二分类，y_true 为 {-1, 1}）
pub fn hinge_loss(y_true : Array[Double], scores : Array[Double]) -> Double {
  if y_true.length() != scores.length() {
    abort("长度不一致")
  }
  let mut sum = 0.0
  for i = 0; i < y_true.length(); i = i + 1 {
    let y = y_true[i]
    if not(y == 1.0 || y == -1.0) {
      abort("标签需为 {-1,1}")
    }
    let m = 1.0 - y * scores[i]
    sum = sum + (if m > 0.0 { m } else { 0.0 })
  }
  sum / y_true.length().to_double()
}

///|
/// L2 正则 (weights 平方和)
pub fn l2_penalty(params : Array[Double], alpha : Double) -> Double {
  if alpha <= 0.0 {
    return 0.0
  }
  let mut sum = 0.0
  for i = 0; i < params.length(); i = i + 1 {
    sum = sum + params[i] * params[i]
  }
  alpha * sum
}

///|
/// L1 正则 (绝对值和)
pub fn l1_penalty(params : Array[Double], alpha : Double) -> Double {
  if alpha <= 0.0 {
    return 0.0
  }
  let mut sum = 0.0
  for i = 0; i < params.length(); i = i + 1 {
    sum = sum + params[i].abs()
  }
  alpha * sum
}
