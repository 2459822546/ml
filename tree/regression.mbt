///|
/// 决策树回归与随机森林回归

///|
pub struct DecisionTreeRegressor {
  mut features : Array[Int]
  mut thresholds : Array[Double]
  mut left : Array[Int]
  mut right : Array[Int]
  mut predictions : Array[Double]
  mut is_leaf : Array[Bool]
  mut feature_importances : Array[Double]
  mut split_seed : Int
  mut n_features : Int
  mut is_fitted : Bool
  max_depth : Int
  min_samples_split : Int
  min_samples_leaf : Int
  random_splits : Bool
}

///|
pub fn DecisionTreeRegressor::new(
  max_depth? : Int = 5,
  min_samples_split? : Int = 2,
  min_samples_leaf? : Int = 1,
  random_splits? : Bool = false,
  seed? : Int = 42,
) -> DecisionTreeRegressor {
  if max_depth <= 0 {
    abort("max_depth 必须大于 0")
  }
  if min_samples_split < 2 {
    abort("min_samples_split 至少为 2")
  }
  if min_samples_leaf < 1 {
    abort("min_samples_leaf 至少为 1")
  }
  {
    features: [],
    thresholds: [],
    left: [],
    right: [],
    predictions: [],
    is_leaf: [],
    feature_importances: [],
    split_seed: seed,
    n_features: 0,
    is_fitted: false,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    random_splits,
  }
}

///|
fn DecisionTreeRegressor::create_node(
  self : DecisionTreeRegressor,
  feature : Int,
  threshold : Double,
  prediction : Double,
  is_leaf_node : Bool,
) -> Int {
  self.features.push(feature)
  self.thresholds.push(threshold)
  self.left.push(-1)
  self.right.push(-1)
  self.predictions.push(prediction)
  self.is_leaf.push(is_leaf_node)
  self.features.length() - 1
}

///|
fn squared_error(targets : Array[Double], mean : Double) -> Double {
  let mut sum = 0.0
  for i = 0; i < targets.length(); i = i + 1 {
    let diff = targets[i] - mean
    sum = sum + diff * diff
  }
  sum
}

///|
fn DecisionTreeRegressor::best_split(
  self : DecisionTreeRegressor,
  x : @math.Matrix,
  y : @math.Vector,
  samples : Array[Int],
) -> (Bool, Int, Double, Array[Int], Array[Int], Double) {
  let n_samples = samples.length()
  let n_features = self.n_features
  let use_random = self.random_splits
  let mut best_feature = -1
  let mut best_threshold = 0.0
  let mut best_error = @double.infinity
  let mut best_left = []
  let mut best_right = []
  for feature = 0; feature < n_features; feature = feature + 1 {
    let idx = Array::make(n_samples, 0)
    for i = 0; i < n_samples; i = i + 1 {
      idx[i] = samples[i]
    }
    idx.sort_by(fn(a, b) {
      let va = x.get(a, feature)
      let vb = x.get(b, feature)
      if va < vb {
        -1
      } else if va > vb {
        1
      } else {
        0
      }
    })
    let vals = Array::make(n_samples, 0.0)
    for i = 0; i < n_samples; i = i + 1 {
      vals[i] = y.get(idx[i])
    }
    let marks = Array::make(n_samples, false)
    if use_random && n_samples > 1 {
      let limit = if n_samples - 1 < 8 { n_samples - 1 } else { 8 }
      for c = 0; c < limit; c = c + 1 {
        self.split_seed = lcg_next(self.split_seed)
        let pos = self.split_seed % (n_samples - 1)
        marks[pos] = true
      }
    }
    let mut left_sum = 0.0
    let mut right_sum = 0.0
    for i = 0; i < n_samples; i = i + 1 {
      right_sum = right_sum + vals[i]
    }
    for i = 0; i < n_samples - 1; i = i + 1 {
      left_sum = left_sum + vals[i]
      right_sum = right_sum - vals[i]
      let v = x.get(idx[i], feature)
      let v_next = x.get(idx[i + 1], feature)
      let should_check = if use_random { marks[i] } else { true }
      if v == v_next || not(should_check) {
        continue
      }
      let left_size = i + 1
      let right_size = n_samples - left_size
      if left_size < self.min_samples_leaf || right_size < self.min_samples_leaf {
        continue
      }
      let left_mean = left_sum / left_size.to_double()
      let right_mean = right_sum / right_size.to_double()
      let left_targets = Array::make(left_size, 0.0)
      let right_targets = Array::make(right_size, 0.0)
      for k = 0; k < left_size; k = k + 1 {
        left_targets[k] = vals[k]
      }
      for k = 0; k < right_size; k = k + 1 {
        right_targets[k] = vals[left_size + k]
      }
      let error = squared_error(left_targets, left_mean) +
        squared_error(right_targets, right_mean)
      if error < best_error {
        best_error = error
        best_feature = feature
        best_threshold = (v + v_next) / 2.0
        best_left = Array::make(left_size, 0)
        best_right = Array::make(right_size, 0)
        for k = 0; k < left_size; k = k + 1 {
          best_left[k] = idx[k]
        }
        for k = 0; k < right_size; k = k + 1 {
          best_right[k] = idx[left_size + k]
        }
      }
    }
  }
  if best_feature == -1 {
    (false, -1, 0.0, [], [], 0.0)
  } else {
    (true, best_feature, best_threshold, best_left, best_right, best_error)
  }
}

///|
pub fn DecisionTreeRegressor::fit(
  self : DecisionTreeRegressor,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.n_features = n_features
  self.features = []
  self.thresholds = []
  self.left = []
  self.right = []
  self.predictions = []
  self.is_leaf = []
  self.feature_importances = Array::make(n_features, 0.0)
  let root_pred = y.mean()
  let root = self.create_node(0, 0.0, root_pred, false)
  let stack = Array::make(1, (root, [], 0))
  let root_samples = Array::make(n_samples, 0)
  for i = 0; i < n_samples; i = i + 1 {
    root_samples[i] = i
  }
  stack[0] = (root, root_samples, 0)
  while stack.length() > 0 {
    match stack.pop() {
      Some(last) => {
        let node_idx = last.0
        let samples = last.1
        let depth = last.2
        let mut sum_targets = 0.0
        for i = 0; i < samples.length(); i = i + 1 {
          sum_targets = sum_targets + y.get(samples[i])
        }
        let mean_val = sum_targets / samples.length().to_double()
        self.predictions[node_idx] = mean_val
        if depth >= self.max_depth || samples.length() < self.min_samples_split {
          self.is_leaf[node_idx] = true
          continue
        }
        let (ok, best_feature, threshold, left_idxes, right_idxes, _) = self.best_split(
          x, y, samples,
        )
        if not(ok) {
          self.is_leaf[node_idx] = true
          continue
        }
        self.feature_importances[best_feature] = self.feature_importances[best_feature] +
          1.0
        let left_node = self.create_node(0, 0.0, mean_val, false)
        let right_node = self.create_node(0, 0.0, mean_val, false)
        self.features[node_idx] = best_feature
        self.thresholds[node_idx] = threshold
        self.left[node_idx] = left_node
        self.right[node_idx] = right_node
        self.left[left_node] = -1
        self.right[left_node] = -1
        self.left[right_node] = -1
        self.right[right_node] = -1
        stack.push((right_node, right_idxes, depth + 1))
        stack.push((left_node, left_idxes, depth + 1))
      }
      None => break
    }
  }
  let mut total_imp = 0.0
  for i = 0; i < self.feature_importances.length(); i = i + 1 {
    total_imp = total_imp + self.feature_importances[i]
  }
  if total_imp > 0.0 {
    for i = 0; i < self.feature_importances.length(); i = i + 1 {
      self.feature_importances[i] = self.feature_importances[i] / total_imp
    }
  }
  self.is_fitted = true
}

///|
pub fn DecisionTreeRegressor::predict(
  self : DecisionTreeRegressor,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let mut node = 0
    while not(self.is_leaf[node]) {
      let feature = self.features[node]
      let threshold = self.thresholds[node]
      let val = x.get(i, feature)
      node = if val <= threshold { self.left[node] } else { self.right[node] }
      if node == -1 {
        break
      }
    }
    out[i] = self.predictions[node]
  }
  @math.Vector::new(out)
}

///|
pub struct RandomForestRegressor {
  n_estimators : Int
  max_depth : Int
  min_samples_split : Int
  min_samples_leaf : Int
  max_features : Int
  seed : Int
  mut trees : Array[DecisionTreeRegressor]
  mut feature_subsets : Array[Array[Int]]
  mut n_features : Int
  mut is_fitted : Bool
}

///|
pub fn RandomForestRegressor::new(
  n_estimators? : Int = 5,
  max_depth? : Int = 5,
  min_samples_split? : Int = 2,
  min_samples_leaf? : Int = 1,
  max_features? : Int = 0,
  seed? : Int = 42,
) -> RandomForestRegressor {
  if n_estimators <= 0 {
    abort("n_estimators 必须大于 0")
  }
  if max_depth <= 0 {
    abort("max_depth 必须大于 0")
  }
  {
    n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
    seed,
    trees: [],
    feature_subsets: [],
    n_features: 0,
    is_fitted: false,
  }
}

///|
pub fn RandomForestRegressor::fit(
  self : RandomForestRegressor,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.n_features = n_features
  self.trees = Array::make(self.n_estimators, DecisionTreeRegressor::new())
  self.feature_subsets = Array::make(self.n_estimators, [])
  let mut seed = self.seed
  for t = 0; t < self.n_estimators; t = t + 1 {
    seed = lcg_next(seed)
    let cols = select_feature_subset(n_features, self.max_features, seed)
    self.feature_subsets[t] = cols
    let sub_x = slice_matrix_cols(x, cols)
    let tree = DecisionTreeRegressor::new(
      max_depth=self.max_depth,
      min_samples_split=self.min_samples_split,
      min_samples_leaf=self.min_samples_leaf,
    )
    tree.fit(sub_x, y)
    self.trees[t] = tree
  }
  self.is_fitted = true
}

///|
pub fn RandomForestRegressor::predict(
  self : RandomForestRegressor,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let mut sum = 0.0
    for t = 0; t < self.trees.length(); t = t + 1 {
      let cols = self.feature_subsets[t]
      let sub_x = slice_matrix_cols(x, cols)
      sum = sum + self.trees[t].predict(sub_x).get(i)
    }
    out[i] = sum / self.trees.length().to_double()
  }
  @math.Vector::new(out)
}

///|
pub fn RandomForestRegressor::feature_importances(
  self : RandomForestRegressor,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let importances = Array::make(self.n_features, 0.0)
  for t = 0; t < self.trees.length(); t = t + 1 {
    let cols = self.feature_subsets[t]
    let tree_imp = self.trees[t].feature_importances
    for j = 0; j < cols.length(); j = j + 1 {
      importances[cols[j]] = importances[cols[j]] + tree_imp[j]
    }
  }
  for j = 0; j < importances.length(); j = j + 1 {
    importances[j] = importances[j] / self.trees.length().to_double()
  }
  @math.Vector::new(importances)
}
