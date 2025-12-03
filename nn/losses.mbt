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
