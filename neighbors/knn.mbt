///|
/// K 最近邻（KNN）分类与回归，使用简单的欧氏距离全量扫描

///|
fn ensure_same_feature(x : @math.Matrix, y : @math.Matrix) -> (Int, Int, Int) {
  let (n_samples, n_features) = x.shape()
  let (other_rows, other_features) = y.shape()
  if n_features != other_features {
    abort("特征维度不一致")
  }
  (n_samples, n_features, other_rows)
}

///|
fn distance(a : @math.Vector, b : @math.Vector) -> Double {
  let (n_a, n_b) = (a.size(), b.size())
  if n_a != n_b {
    abort("向量长度不一致")
  }
  let mut sum = 0.0
  for i = 0; i < n_a; i = i + 1 {
    let diff = a.get(i) - b.get(i)
    sum = sum + diff * diff
  }
  sum.sqrt()
}

///|
fn top_k_indices(distances : Array[Double], k : Int) -> Array[Int] {
  let idx = Array::make(distances.length(), 0)
  for i = 0; i < distances.length(); i = i + 1 {
    idx[i] = i
  }
  idx.sort_by(fn(a, b) {
    let da = distances[a]
    let db = distances[b]
    if da < db {
      -1
    } else if da > db {
      1
    } else {
      0
    }
  })
  let result = Array::make(k, 0)
  for i = 0; i < k; i = i + 1 {
    result[i] = idx[i]
  }
  result
}

///|
fn majority_vote(labels : @math.Vector, neighbors : Array[Int]) -> Double {
  let uniq = []
  let counts = []
  for i = 0; i < neighbors.length(); i = i + 1 {
    let val = labels.get(neighbors[i])
    let mut found = -1
    for j = 0; j < uniq.length(); j = j + 1 {
      if uniq[j] == val {
        found = j
        break
      }
    }
    if found == -1 {
      uniq.push(val)
      counts.push(1)
    } else {
      counts[found] = counts[found] + 1
    }
  }
  let mut best_idx = 0
  let mut best_count = -1
  for i = 0; i < counts.length(); i = i + 1 {
    if counts[i] > best_count {
      best_count = counts[i]
      best_idx = i
    } else if counts[i] == best_count && uniq[i] < uniq[best_idx] {
      best_idx = i
    }
  }
  uniq[best_idx]
}

///|
fn mean_value(labels : @math.Vector, neighbors : Array[Int]) -> Double {
  if neighbors.length() == 0 {
    abort("至少需要一个邻居")
  }
  let mut total = 0.0
  for i = 0; i < neighbors.length(); i = i + 1 {
    total = total + labels.get(neighbors[i])
  }
  total / neighbors.length().to_double()
}

///|
/// KNN 分类器
pub struct KNeighborsClassifier {
  k : Int
  mut train_x : @math.Matrix
  mut train_y : @math.Vector
  mut is_fitted : Bool
}

///|
pub fn KNeighborsClassifier::new(k? : Int = 5) -> KNeighborsClassifier {
  if k <= 0 {
    abort("k 必须大于 0")
  }
  {
    k,
    train_x: @math.Matrix::ones(1, 1),
    train_y: @math.Vector::zeros(1),
    is_fitted: false,
  }
}

///|
/// 拟合训练数据
pub fn KNeighborsClassifier::fit(
  self : KNeighborsClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, _) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不一致")
  }
  if self.k > n_samples {
    abort("k 不能大于样本数")
  }
  self.train_x = x
  self.train_y = y
  self.is_fitted = true
}

///|
/// 预测标签
pub fn KNeighborsClassifier::predict(
  self : KNeighborsClassifier,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _, _) = ensure_same_feature(x, self.train_x)
  let preds = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let query = x.get_row(i)
    let distances = Array::make(self.train_y.size(), 0.0)
    for j = 0; j < self.train_y.size(); j = j + 1 {
      let row = self.train_x.get_row(j)
      distances[j] = distance(query, row)
    }
    let neighbors = top_k_indices(distances, self.k)
    preds[i] = majority_vote(self.train_y, neighbors)
  }
  @math.Vector::new(preds)
}

///|
/// KNN 回归器
pub struct KNeighborsRegressor {
  k : Int
  mut train_x : @math.Matrix
  mut train_y : @math.Vector
  mut is_fitted : Bool
}

///|
pub fn KNeighborsRegressor::new(k? : Int = 5) -> KNeighborsRegressor {
  if k <= 0 {
    abort("k 必须大于 0")
  }
  {
    k,
    train_x: @math.Matrix::ones(1, 1),
    train_y: @math.Vector::zeros(1),
    is_fitted: false,
  }
}

///|
/// 拟合训练数据
pub fn KNeighborsRegressor::fit(
  self : KNeighborsRegressor,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, _) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不一致")
  }
  if self.k > n_samples {
    abort("k 不能大于样本数")
  }
  self.train_x = x
  self.train_y = y
  self.is_fitted = true
}

///|
/// 预测数值
pub fn KNeighborsRegressor::predict(
  self : KNeighborsRegressor,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _, _) = ensure_same_feature(x, self.train_x)
  let preds = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let query = x.get_row(i)
    let distances = Array::make(self.train_y.size(), 0.0)
    for j = 0; j < self.train_y.size(); j = j + 1 {
      let row = self.train_x.get_row(j)
      distances[j] = distance(query, row)
    }
    let neighbors = top_k_indices(distances, self.k)
    preds[i] = mean_value(self.train_y, neighbors)
  }
  @math.Vector::new(preds)
}
