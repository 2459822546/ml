///|
/// 线性分类模型
/// 包括逻辑回归、感知机和随机梯度下降分类器

///|
/// 逻辑回归 (Logistic Regression)
pub struct LogisticRegression {
  mut weights : @math.Vector
  mut intercept : Double
  mut is_fitted : Bool
  learning_rate : Double
  max_iter : Int
  tol : Double
}

///|
/// 感知机 (Perceptron)
pub struct Perceptron {
  mut weights : @math.Vector
  mut intercept : Double
  mut is_fitted : Bool
  learning_rate : Double
  max_iter : Int
}

///|
/// 随机梯度下降分类器 (SGD Classifier)
pub struct SGDClassifier {
  mut weights : @math.Vector
  mut intercept : Double
  mut is_fitted : Bool
  learning_rate : Double
  max_iter : Int
  alpha : Double // L2 正则化系数
  loss : LossFunction
}

///|
/// 损失函数类型
pub(all) enum LossFunction {
  Hinge // SVM 损失
  Log // 逻辑损失
  Perceptron // 感知机损失
}

///|
/// ========== 逻辑回归 ==========

///|
pub fn LogisticRegression::new(
  learning_rate : Double,
  max_iter : Int,
  tol : Double,
) -> LogisticRegression {
  {
    weights: @math.Vector::zeros(0),
    intercept: 0.0,
    is_fitted: false,
    learning_rate,
    max_iter,
    tol,
  }
}

///|
/// Sigmoid 函数
fn sigmoid(z : Double) -> Double {
  1.0 / (1.0 + @moonbitlang/core/math.exp(-z))
}

///|
/// 训练逻辑回归模型（使用梯度下降）
pub fn fit(
  self : LogisticRegression,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }

  // 初始化权重
  self.weights = @math.Vector::zeros(n_features)
  self.intercept = 0.0

  // 梯度下降
  for _iter = 0; _iter < self.max_iter; _iter = _iter + 1 {
    // 计算预测值
    let predictions = Array::make(n_samples, 0.0)
    for i = 0; i < n_samples; i = i + 1 {
      let row = x.get_row(i)
      let z = self.weights.dot(row) + self.intercept
      predictions[i] = sigmoid(z)
    }

    // 计算梯度
    let grad_w = @math.Vector::zeros(n_features)
    let mut grad_b = 0.0
    for i = 0; i < n_samples; i = i + 1 {
      let error = predictions[i] - y.get(i)
      grad_b = grad_b + error
      let row = x.get_row(i)
      for j = 0; j < n_features; j = j + 1 {
        grad_w.set(j, grad_w.get(j) + error * row.get(j))
      }
    }

    // 更新权重
    for j = 0; j < n_features; j = j + 1 {
      let w_j = self.weights.get(j) -
        self.learning_rate * grad_w.get(j) / n_samples.to_double()
      self.weights.set(j, w_j)
    }
    self.intercept = self.intercept -
      self.learning_rate * grad_b / n_samples.to_double()

    // 检查收敛（简化版）
    if grad_w.norm_l2() < self.tol {
      break
    }
  }
  self.is_fitted = true
}

///|
/// 预测概率
pub fn predict_proba(
  self : LogisticRegression,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let probas = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    let z = self.weights.dot(row) + self.intercept
    probas[i] = sigmoid(z)
  }
  @math.Vector::new(probas)
}

///|
/// 预测类别（阈值为0.5）
pub fn predict(self : LogisticRegression, x : @math.Matrix) -> @math.Vector {
  let probas = self.predict_proba(x)
  let predictions = Array::make(probas.size(), 0.0)
  for i = 0; i < probas.size(); i = i + 1 {
    predictions[i] = if probas.get(i) >= 0.5 { 1.0 } else { 0.0 }
  }
  @math.Vector::new(predictions)
}

///|
/// 计算准确率
pub fn score(
  self : LogisticRegression,
  x : @math.Matrix,
  y : @math.Vector,
) -> Double {
  let y_pred = self.predict(x)
  accuracy_score(y, y_pred)
}

///|
/// ========== 感知机 ==========

///|
pub fn Perceptron::new(learning_rate : Double, max_iter : Int) -> Perceptron {
  {
    weights: @math.Vector::zeros(0),
    intercept: 0.0,
    is_fitted: false,
    learning_rate,
    max_iter,
  }
}

///|
/// 训练感知机
pub fn fit(self : Perceptron, x : @math.Matrix, y : @math.Vector) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }

  // 初始化权重
  self.weights = @math.Vector::zeros(n_features)
  self.intercept = 0.0

  // 感知机学习算法
  for _epoch = 0; _epoch < self.max_iter; _epoch = _epoch + 1 {
    let mut errors = 0
    for i = 0; i < n_samples; i = i + 1 {
      let row = x.get_row(i)
      let activation = self.weights.dot(row) + self.intercept
      let prediction = if activation >= 0.0 { 1.0 } else { 0.0 }
      let y_i = y.get(i)
      if prediction != y_i {
        errors = errors + 1

        // 更新权重
        let update = self.learning_rate * (y_i - prediction)
        for j = 0; j < n_features; j = j + 1 {
          let w_j = self.weights.get(j) + update * row.get(j)
          self.weights.set(j, w_j)
        }
        self.intercept = self.intercept + update
      }
    }

    // 如果没有错误，提前停止
    if errors == 0 {
      break
    }
  }
  self.is_fitted = true
}

///|
/// 预测
pub fn predict(self : Perceptron, x : @math.Matrix) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let predictions = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    let activation = self.weights.dot(row) + self.intercept
    predictions[i] = if activation >= 0.0 { 1.0 } else { 0.0 }
  }
  @math.Vector::new(predictions)
}

///|
pub fn score(self : Perceptron, x : @math.Matrix, y : @math.Vector) -> Double {
  let y_pred = self.predict(x)
  accuracy_score(y, y_pred)
}

///|
/// ========== 随机梯度下降分类器 ==========

///|
pub fn SGDClassifier::new(
  learning_rate : Double,
  max_iter : Int,
  alpha : Double,
  loss : LossFunction,
) -> SGDClassifier {
  {
    weights: @math.Vector::zeros(0),
    intercept: 0.0,
    is_fitted: false,
    learning_rate,
    max_iter,
    alpha,
    loss,
  }
}

///|
/// 训练 SGD 分类器
pub fn fit(self : SGDClassifier, x : @math.Matrix, y : @math.Vector) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }

  // 初始化权重
  self.weights = @math.Vector::zeros(n_features)
  self.intercept = 0.0

  // 随机梯度下降
  for _epoch = 0; _epoch < self.max_iter; _epoch = _epoch + 1 {
    for i = 0; i < n_samples; i = i + 1 {
      let row = x.get_row(i)
      let y_i = y.get(i)
      let activation = self.weights.dot(row) + self.intercept

      // 根据损失函数计算梯度
      let (grad_loss, should_update) = match self.loss {
        Hinge => {
          // SVM hinge loss: max(0, 1 - y * activation)
          let margin = y_i * activation
          if margin < 1.0 {
            (-y_i, true)
          } else {
            (0.0, false)
          }
        }
        Log => {
          // Logistic loss
          let pred = sigmoid(activation)
          (pred - y_i, true)
        }
        Perceptron => {
          // Perceptron loss
          let pred = if activation >= 0.0 { 1.0 } else { 0.0 }
          if pred != y_i {
            (pred - y_i, true)
          } else {
            (0.0, false)
          }
        }
      }
      if should_update {
        // 更新权重（带 L2 正则化）
        for j = 0; j < n_features; j = j + 1 {
          let w_j = self.weights.get(j)
          let grad = grad_loss * row.get(j) + self.alpha * w_j
          self.weights.set(j, w_j - self.learning_rate * grad)
        }
        self.intercept = self.intercept - self.learning_rate * grad_loss
      } else if self.alpha > 0.0 {
        // 仅应用正则化
        for j = 0; j < n_features; j = j + 1 {
          let w_j = self.weights.get(j)
          self.weights.set(j, w_j * (1.0 - self.learning_rate * self.alpha))
        }
      }
    }
  }
  self.is_fitted = true
}

///|
/// 预测
pub fn predict(self : SGDClassifier, x : @math.Matrix) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let predictions = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    let activation = self.weights.dot(row) + self.intercept
    predictions[i] = match self.loss {
      Log =>
        // 逻辑回归风格预测
        if sigmoid(activation) >= 0.5 {
          1.0
        } else {
          0.0
        }
      _ =>
        // 线性决策边界
        if activation >= 0.0 {
          1.0
        } else {
          0.0
        }
    }
  }
  @math.Vector::new(predictions)
}

///|
pub fn score(
  self : SGDClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Double {
  let y_pred = self.predict(x)
  accuracy_score(y, y_pred)
}

///|
/// ========== 辅助函数 ==========

///|
/// 计算准确率
fn accuracy_score(y_true : @math.Vector, y_pred : @math.Vector) -> Double {
  if y_true.size() != y_pred.size() {
    abort("y_true 和 y_pred 长度不匹配")
  }
  let mut correct = 0
  for i = 0; i < y_true.size(); i = i + 1 {
    if y_true.get(i) == y_pred.get(i) {
      correct = correct + 1
    }
  }
  correct.to_double() / y_true.size().to_double()
}
