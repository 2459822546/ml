///|
/// 朴素贝叶斯分类器：高斯、多项式、伯努利

///|
pub struct GaussianNB {
  mut class_priors : Array[Double]
  mut class_means : Array[Array[Double]]
  mut class_vars : Array[Array[Double]]
  mut classes : Array[Double]
  mut is_fitted : Bool
}

///|
pub fn GaussianNB::new() -> GaussianNB {
  {
    class_priors: [],
    class_means: [],
    class_vars: [],
    classes: [],
    is_fitted: false,
  }
}

///|
fn nb_unique_labels(y : @math.Vector) -> Array[Double] {
  let uniq = []
  for i = 0; i < y.size(); i = i + 1 {
    let v = y.get(i)
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
pub fn GaussianNB::fit(
  self : GaussianNB,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.classes = nb_unique_labels(y)
  let n_classes = self.classes.length()
  self.class_priors = Array::make(n_classes, 0.0)
  self.class_means = Array::make(n_classes, [])
  self.class_vars = Array::make(n_classes, [])
  for c = 0; c < n_classes; c = c + 1 {
    let cls = self.classes[c]
    let mut count = 0
    let sum = Array::make(n_features, 0.0)
    let sum_sq = Array::make(n_features, 0.0)
    for i = 0; i < n_samples; i = i + 1 {
      if y.get(i) == cls {
        count = count + 1
        let row = x.get_row(i)
        for j = 0; j < n_features; j = j + 1 {
          let v = row.get(j)
          sum[j] = sum[j] + v
          sum_sq[j] = sum_sq[j] + v * v
        }
      }
    }
    if count == 0 {
      abort("类别计数为 0")
    }
    self.class_priors[c] = count.to_double() / n_samples.to_double()
    let means = Array::make(n_features, 0.0)
    let vars = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      let mean = sum[j] / count.to_double()
      let variance = sum_sq[j] / count.to_double() - mean * mean
      means[j] = mean
      vars[j] = if variance <= 1.0e-9 { 1.0e-9 } else { variance }
    }
    self.class_means[c] = means
    self.class_vars[c] = vars
  }
  self.is_fitted = true
}

///|
fn gaussian_log_prob(
  x : @math.Vector,
  mean : Array[Double],
  variance : Array[Double],
) -> Double {
  let n = x.size()
  let mut logp = 0.0
  for j = 0; j < n; j = j + 1 {
    let diff = x.get(j) - mean[j]
    logp = logp -
      0.5 * (diff * diff / variance[j] + @moonbitlang/core/math.ln(variance[j]))
  }
  logp
}

///|
pub fn GaussianNB::predict(self : GaussianNB, x : @math.Matrix) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    let mut best = 0
    let mut best_score = -@double.infinity
    for c = 0; c < self.classes.length(); c = c + 1 {
      let score = @moonbitlang/core/math.ln(self.class_priors[c]) +
        gaussian_log_prob(row, self.class_means[c], self.class_vars[c])
      if score > best_score {
        best_score = score
        best = c
      }
    }
    out[i] = self.classes[best]
  }
  @math.Vector::new(out)
}

///|
pub fn GaussianNB::score(
  self : GaussianNB,
  x : @math.Matrix,
  y : @math.Vector,
) -> Double {
  let pred = self.predict(x)
  let mut correct = 0
  for i = 0; i < pred.size(); i = i + 1 {
    if pred.get(i) == y.get(i) {
      correct = correct + 1
    }
  }
  correct.to_double() / pred.size().to_double()
}

///|
pub struct MultinomialNB {
  mut class_priors : Array[Double]
  mut feature_log_prob : Array[Array[Double]]
  mut classes : Array[Double]
  alpha : Double
  mut is_fitted : Bool
}

///|
pub fn MultinomialNB::new(alpha? : Double = 1.0) -> MultinomialNB {
  if alpha < 0.0 {
    abort("alpha 不能为负")
  }
  {
    class_priors: [],
    feature_log_prob: [],
    classes: [],
    alpha,
    is_fitted: false,
  }
}

///|
pub fn MultinomialNB::fit(
  self : MultinomialNB,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.classes = nb_unique_labels(y)
  let n_classes = self.classes.length()
  self.class_priors = Array::make(n_classes, 0.0)
  self.feature_log_prob = Array::make(n_classes, [])
  for c = 0; c < n_classes; c = c + 1 {
    let cls = self.classes[c]
    let mut class_count = 0.0
    let feature_counts = Array::make(n_features, self.alpha)
    for i = 0; i < n_samples; i = i + 1 {
      if y.get(i) == cls {
        class_count = class_count + 1.0
        let row = x.get_row(i)
        for j = 0; j < n_features; j = j + 1 {
          feature_counts[j] = feature_counts[j] + row.get(j)
        }
      }
    }
    if class_count == 0.0 {
      abort("类别计数为 0")
    }
    let mut total = 0.0
    for j = 0; j < n_features; j = j + 1 {
      total = total + feature_counts[j]
    }
    let log_probs = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      log_probs[j] = @moonbitlang/core/math.ln(feature_counts[j] / total)
    }
    self.class_priors[c] = class_count / n_samples.to_double()
    self.feature_log_prob[c] = log_probs
  }
  self.is_fitted = true
}

///|
pub fn MultinomialNB::predict(
  self : MultinomialNB,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    let mut best = 0
    let mut best_score = -@double.infinity
    for c = 0; c < self.classes.length(); c = c + 1 {
      let mut score = @moonbitlang/core/math.ln(self.class_priors[c])
      let log_probs = self.feature_log_prob[c]
      for j = 0; j < row.size(); j = j + 1 {
        score = score + row.get(j) * log_probs[j]
      }
      if score > best_score {
        best_score = score
        best = c
      }
    }
    out[i] = self.classes[best]
  }
  @math.Vector::new(out)
}

///|
pub fn MultinomialNB::score(
  self : MultinomialNB,
  x : @math.Matrix,
  y : @math.Vector,
) -> Double {
  let pred = self.predict(x)
  let mut correct = 0
  for i = 0; i < pred.size(); i = i + 1 {
    if pred.get(i) == y.get(i) {
      correct = correct + 1
    }
  }
  correct.to_double() / pred.size().to_double()
}

///|
pub struct BernoulliNB {
  mut class_priors : Array[Double]
  mut feature_log_prob : Array[Array[Double]]
  mut feature_log_prob_neg : Array[Array[Double]]
  mut classes : Array[Double]
  alpha : Double
  mut is_fitted : Bool
}

///|
pub fn BernoulliNB::new(alpha? : Double = 1.0) -> BernoulliNB {
  if alpha < 0.0 {
    abort("alpha 不能为负")
  }
  {
    class_priors: [],
    feature_log_prob: [],
    feature_log_prob_neg: [],
    classes: [],
    alpha,
    is_fitted: false,
  }
}

///|
pub fn BernoulliNB::fit(
  self : BernoulliNB,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.classes = nb_unique_labels(y)
  let n_classes = self.classes.length()
  self.class_priors = Array::make(n_classes, 0.0)
  self.feature_log_prob = Array::make(n_classes, [])
  self.feature_log_prob_neg = Array::make(n_classes, [])
  for c = 0; c < n_classes; c = c + 1 {
    let cls = self.classes[c]
    let mut class_count = 0.0
    let feature_counts = Array::make(n_features, self.alpha)
    for i = 0; i < n_samples; i = i + 1 {
      if y.get(i) == cls {
        class_count = class_count + 1.0
        let row = x.get_row(i)
        for j = 0; j < n_features; j = j + 1 {
          let inc = if row.get(j) > 0.0 { 1.0 } else { 0.0 }
          feature_counts[j] = feature_counts[j] + inc
        }
      }
    }
    if class_count == 0.0 {
      abort("类别计数为 0")
    }
    let log_probs = Array::make(n_features, 0.0)
    let log_probs_neg = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      let p = feature_counts[j] / (class_count + 2.0 * self.alpha)
      log_probs[j] = @moonbitlang/core/math.ln(p)
      log_probs_neg[j] = @moonbitlang/core/math.ln(1.0 - p)
    }
    self.class_priors[c] = class_count / n_samples.to_double()
    self.feature_log_prob[c] = log_probs
    self.feature_log_prob_neg[c] = log_probs_neg
  }
  self.is_fitted = true
}

///|
pub fn BernoulliNB::predict(
  self : BernoulliNB,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    let mut best = 0
    let mut best_score = -@double.infinity
    for c = 0; c < self.classes.length(); c = c + 1 {
      let mut score = @moonbitlang/core/math.ln(self.class_priors[c])
      let lp = self.feature_log_prob[c]
      let ln = self.feature_log_prob_neg[c]
      for j = 0; j < row.size(); j = j + 1 {
        if row.get(j) > 0.0 {
          score = score + lp[j]
        } else {
          score = score + ln[j]
        }
      }
      if score > best_score {
        best_score = score
        best = c
      }
    }
    out[i] = self.classes[best]
  }
  @math.Vector::new(out)
}

///|
pub fn BernoulliNB::score(
  self : BernoulliNB,
  x : @math.Matrix,
  y : @math.Vector,
) -> Double {
  let pred = self.predict(x)
  let mut correct = 0
  for i = 0; i < pred.size(); i = i + 1 {
    if pred.get(i) == y.get(i) {
      correct = correct + 1
    }
  }
  correct.to_double() / pred.size().to_double()
}
