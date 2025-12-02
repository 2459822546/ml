///|
/// One-vs-Rest 多分类封装（使用 LogisticRegression 作为基分类器）

///|
pub struct OneVsRestClassifier {
  learning_rate : Double
  max_iter : Int
  tol : Double
  mut classes : Array[Double]
  mut models : Array[LogisticRegression]
  mut is_fitted : Bool
}

///|
pub fn OneVsRestClassifier::new(
  learning_rate? : Double = 0.1,
  max_iter? : Int = 200,
  tol? : Double = 1.0e-4,
) -> OneVsRestClassifier {
  if learning_rate <= 0.0 {
    abort("learning_rate 必须大于 0")
  }
  if max_iter <= 0 {
    abort("max_iter 必须大于 0")
  }
  if tol <= 0.0 {
    abort("tol 必须大于 0")
  }
  { learning_rate, max_iter, tol, classes: [], models: [], is_fitted: false }
}

///|
fn unique_labels(y : @math.Vector) -> Array[Double] {
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
pub fn OneVsRestClassifier::fit(
  self : OneVsRestClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, _) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.classes = unique_labels(y)
  if self.classes.length() < 2 {
    abort("至少需要两个类别")
  }
  self.models = Array::make(
    self.classes.length(),
    LogisticRegression::new(self.learning_rate, self.max_iter, self.tol),
  )
  for c = 0; c < self.classes.length(); c = c + 1 {
    let target = self.classes[c]
    let bin = Array::make(n_samples, 0.0)
    for i = 0; i < n_samples; i = i + 1 {
      bin[i] = if y.get(i) == target { 1.0 } else { 0.0 }
    }
    let y_bin = @math.Vector::new(bin)
    self.models[c] = LogisticRegression::new(
      self.learning_rate,
      self.max_iter,
      self.tol,
    )
    self.models[c].fit(x, y_bin)
  }
  self.is_fitted = true
}

///|
pub fn OneVsRestClassifier::predict(
  self : OneVsRestClassifier,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let probs = Array::make(self.classes.length(), @math.Vector::new([]))
  for c = 0; c < self.classes.length(); c = c + 1 {
    probs[c] = self.models[c].predict_proba(x)
  }
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let mut best = 0
    let mut best_prob = -1.0
    for c = 0; c < self.classes.length(); c = c + 1 {
      let p = probs[c].get(i)
      if p > best_prob {
        best_prob = p
        best = c
      }
    }
    out[i] = self.classes[best]
  }
  @math.Vector::new(out)
}

///|
pub fn OneVsRestClassifier::score(
  self : OneVsRestClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Double {
  let pred = self.predict(x)
  if pred.size() != y.size() {
    abort("y 长度不匹配")
  }
  let mut correct = 0
  for i = 0; i < pred.size(); i = i + 1 {
    if pred.get(i) == y.get(i) {
      correct = correct + 1
    }
  }
  correct.to_double() / pred.size().to_double()
}
