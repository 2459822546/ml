///|
/// 线性 SVM (hinge loss) 分类器

///|
pub struct LinearSVM {
  mut weights : @math.Vector
  mut intercept : Double
  mut is_fitted : Bool
  c : Double
  learning_rate : Double
  max_iter : Int
}

///|
pub fn LinearSVM::new(
  c? : Double = 1.0,
  learning_rate? : Double = 0.1,
  max_iter? : Int = 200,
) -> LinearSVM {
  if c <= 0.0 {
    abort("C 必须大于 0")
  }
  if learning_rate <= 0.0 {
    abort("learning_rate 必须大于 0")
  }
  if max_iter <= 0 {
    abort("max_iter 必须大于 0")
  }
  {
    weights: @math.Vector::zeros(0),
    intercept: 0.0,
    is_fitted: false,
    c,
    learning_rate,
    max_iter,
  }
}

///|
pub fn LinearSVM::fit(
  self : LinearSVM,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.weights = @math.Vector::zeros(n_features)
  self.intercept = 0.0
  for iter = 0; iter < self.max_iter; iter = iter + 1 {
    for i = 0; i < n_samples; i = i + 1 {
      let row = x.get_row(i)
      let yi = y.get(i)
      let margin = yi * (self.weights.dot(row) + self.intercept)
      if margin < 1.0 {
        for j = 0; j < n_features; j = j + 1 {
          let wj = self.weights.get(j)
          let grad = wj - self.c * yi * row.get(j)
          self.weights.set(j, wj - self.learning_rate * grad)
        }
        self.intercept = self.intercept - self.learning_rate * (-self.c * yi)
      } else {
        for j = 0; j < n_features; j = j + 1 {
          let wj = self.weights.get(j)
          self.weights.set(j, wj - self.learning_rate * wj)
        }
      }
    }
  }
  self.is_fitted = true
}

///|
pub fn LinearSVM::predict(self : LinearSVM, x : @math.Matrix) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    let margin = self.weights.dot(row) + self.intercept
    out[i] = if margin >= 0.0 { 1.0 } else { -1.0 }
  }
  @math.Vector::new(out)
}

///|
pub fn LinearSVM::score(
  self : LinearSVM,
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
