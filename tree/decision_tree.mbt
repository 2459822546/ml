///|
/// 决策树与随机森林（仅数值特征）

///|
/// 简单决策树分类器（基于 Gini）
pub struct DecisionTreeClassifier {
  max_depth : Int
  min_samples_split : Int
  min_samples_leaf : Int
  random_splits : Bool
  mut features : Array[Int]
  mut thresholds : Array[Double]
  mut left : Array[Int]
  mut right : Array[Int]
  mut predictions : Array[Double]
  mut is_leaf : Array[Bool]
  mut feature_importances : Array[Double]
  mut split_seed : Int
  mut n_features : Int
  mut classes : Array[Double]
  mut is_fitted : Bool
}

///|
pub fn DecisionTreeClassifier::new(
  max_depth? : Int = 5,
  min_samples_split? : Int = 2,
  min_samples_leaf? : Int = 1,
  random_splits? : Bool = false,
  seed? : Int = 42,
) -> DecisionTreeClassifier {
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
    max_depth,
    min_samples_split,
    min_samples_leaf,
    random_splits,
    features: [],
    thresholds: [],
    left: [],
    right: [],
    predictions: [],
    is_leaf: [],
    feature_importances: [],
    split_seed: seed,
    n_features: 0,
    classes: [],
    is_fitted: false,
  }
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
fn label_index(label : Double, classes : Array[Double]) -> Int {
  for i = 0; i < classes.length(); i = i + 1 {
    if classes[i] == label {
      return i
    }
  }
  abort("未知标签")
}

///|
fn gini_from_counts(counts : Array[Int], total : Int) -> Double {
  if total == 0 {
    return 0.0
  }
  let mut sum_sq = 0.0
  let total_d = total.to_double()
  for i = 0; i < counts.length(); i = i + 1 {
    let p = counts[i].to_double() / total_d
    sum_sq = sum_sq + p * p
  }
  1.0 - sum_sq
}

///|
fn majority_label(counts : Array[Int], classes : Array[Double]) -> Double {
  let mut best = 0
  let mut best_count = -1
  for i = 0; i < counts.length(); i = i + 1 {
    if counts[i] > best_count {
      best_count = counts[i]
      best = i
    }
  }
  classes[best]
}

///|
fn DecisionTreeClassifier::create_node(
  self : DecisionTreeClassifier,
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
fn DecisionTreeClassifier::evaluate_best_split(
  self : DecisionTreeClassifier,
  x : @math.Matrix,
  y : @math.Vector,
  samples : Array[Int],
  parent_gini : Double,
) -> (Bool, Int, Double, Array[Int], Array[Int], Double) {
  let n_samples = samples.length()
  let n_features = self.n_features
  let use_random = self.random_splits
  let n_classes = self.classes.length()
  let mut best_feature = -1
  let mut best_threshold = 0.0
  let mut best_gini = @double.infinity
  let mut best_left : Array[Int] = []
  let mut best_right : Array[Int] = []
  for feature = 0; feature < n_features; feature = feature + 1 {
    // 按特征值排序的样本索引
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
    let left_counts = Array::make(n_classes, 0)
    let right_counts = Array::make(n_classes, 0)
    for i = 0; i < n_samples; i = i + 1 {
      let cls = label_index(y.get(idx[i]), self.classes)
      right_counts[cls] = right_counts[cls] + 1
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
    for i = 0; i < n_samples - 1; i = i + 1 {
      let cls = label_index(y.get(idx[i]), self.classes)
      left_counts[cls] = left_counts[cls] + 1
      right_counts[cls] = right_counts[cls] - 1
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
      let g_left = gini_from_counts(left_counts, left_size)
      let g_right = gini_from_counts(right_counts, right_size)
      let weighted = g_left * left_size.to_double() / n_samples.to_double() +
        g_right * right_size.to_double() / n_samples.to_double()
      if weighted < best_gini {
        best_gini = weighted
        best_feature = feature
        best_threshold = (v + v_next) / 2.0
        // 拷贝左右样本
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
    let impurity_decrease = parent_gini - best_gini
    (
      true, best_feature, best_threshold, best_left, best_right, impurity_decrease,
    )
  }
}

///|
pub fn DecisionTreeClassifier::fit(
  self : DecisionTreeClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.features = []
  self.thresholds = []
  self.left = []
  self.right = []
  self.predictions = []
  self.is_leaf = []
  self.n_features = n_features
  self.classes = unique_labels(y)
  self.feature_importances = Array::make(n_features, 0.0)

  // 根节点初始化
  let root = self.create_node(0, 0.0, 0.0, false)

  // 样本计数
  fn counts_for_samples(
    y : @math.Vector,
    samples : Array[Int],
    classes : Array[Double],
  ) -> Array[Int] {
    let counts = Array::make(classes.length(), 0)
    for i = 0; i < samples.length(); i = i + 1 {
      let cls_idx = label_index(y.get(samples[i]), classes)
      counts[cls_idx] = counts[cls_idx] + 1
    }
    counts
  }

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
        let counts = counts_for_samples(y, samples, self.classes)
        let prediction = majority_label(counts, self.classes)
        self.predictions[node_idx] = prediction
        let is_pure = {
          let mut non_zero = 0
          for i = 0; i < counts.length(); i = i + 1 {
            if counts[i] > 0 {
              non_zero = non_zero + 1
            }
          }
          non_zero == 1
        }
        if is_pure ||
          depth >= self.max_depth ||
          samples.length() < self.min_samples_split {
          self.is_leaf[node_idx] = true
          continue
        }
        let parent_gini = gini_from_counts(counts, samples.length())
        let (ok, best_feature, threshold, left_idxes, right_idxes, impurity_gain
        ) = self.evaluate_best_split(x, y, samples, parent_gini)
        if not(ok) {
          self.is_leaf[node_idx] = true
          continue
        }

        // 记录特征重要性
        self.feature_importances[best_feature] = self.feature_importances[best_feature] +
          impurity_gain

        // 建子节点
        let left_node = self.create_node(0, 0.0, prediction, false)
        let right_node = self.create_node(0, 0.0, prediction, false)
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

  // 归一化重要性
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
pub fn DecisionTreeClassifier::predict(
  self : DecisionTreeClassifier,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let preds = Array::make(n_samples, 0.0)
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
    preds[i] = self.predictions[node]
  }
  @math.Vector::new(preds)
}

///|
pub fn DecisionTreeClassifier::score(
  self : DecisionTreeClassifier,
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

///|
pub fn DecisionTreeClassifier::get_feature_importances(
  self : DecisionTreeClassifier,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  @math.Vector::new(self.feature_importances)
}

///|
/// 随机森林分类器（使用简单 LCG bootstrap 和列子集）
pub struct RandomForestClassifier {
  n_estimators : Int
  max_depth : Int
  min_samples_split : Int
  min_samples_leaf : Int
  max_features : Int
  seed : Int
  mut trees : Array[DecisionTreeClassifier]
  mut feature_subsets : Array[Array[Int]]
  mut classes : Array[Double]
  mut n_features : Int
  mut is_fitted : Bool
}

///|
pub fn RandomForestClassifier::new(
  n_estimators? : Int = 5,
  max_depth? : Int = 5,
  min_samples_split? : Int = 2,
  min_samples_leaf? : Int = 1,
  max_features? : Int = 0, // 0 表示 sqrt(n_features)
  seed? : Int = 42,
) -> RandomForestClassifier {
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
    classes: [],
    n_features: 0,
    is_fitted: false,
  }
}

///|
fn lcg_next(seed : Int) -> Int {
  (1103515245 * seed + 12345) & 0x7fffffff
}

///|
fn select_feature_subset(n_features : Int, k : Int, seed : Int) -> Array[Int] {
  let k_use = if k <= 0 || k > n_features {
    let root = n_features.to_double().sqrt().to_int()
    if root == 0 {
      1
    } else {
      root
    }
  } else {
    k
  }
  let cols = Array::make(n_features, 0)
  for i = 0; i < n_features; i = i + 1 {
    cols[i] = i
  }
  // 简单基于种子的偏移选择
  let start = seed % n_features
  let subset = Array::make(k_use, 0)
  for i = 0; i < k_use; i = i + 1 {
    subset[i] = cols[(start + i) % n_features]
  }
  subset
}

///|
fn slice_matrix_cols(x : @math.Matrix, cols : Array[Int]) -> @math.Matrix {
  let (n_samples, _) = x.shape()
  let out = @math.Matrix::zeros(n_samples, cols.length())
  for i = 0; i < n_samples; i = i + 1 {
    let row = x.get_row(i)
    for j = 0; j < cols.length(); j = j + 1 {
      out.set(i, j, row.get(cols[j]))
    }
  }
  out
}

///|
pub fn RandomForestClassifier::fit(
  self : RandomForestClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.classes = unique_labels(y)
  self.n_features = n_features
  self.trees = Array::make(self.n_estimators, DecisionTreeClassifier::new())
  self.feature_subsets = Array::make(self.n_estimators, [])
  let mut seed = self.seed
  for t = 0; t < self.n_estimators; t = t + 1 {
    seed = lcg_next(seed)
    let cols = select_feature_subset(n_features, self.max_features, seed)
    self.feature_subsets[t] = cols
    let boot_x = slice_matrix_cols(x, cols)
    let tree = DecisionTreeClassifier::new(
      max_depth=self.max_depth,
      min_samples_split=self.min_samples_split,
      min_samples_leaf=self.min_samples_leaf,
    )
    tree.fit(boot_x, y)
    self.trees[t] = tree
  }
  self.is_fitted = true
}

///|
pub fn RandomForestClassifier::predict(
  self : RandomForestClassifier,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let votes = Array::make(n_samples, Array::make(self.classes.length(), 0))
  for t = 0; t < self.trees.length(); t = t + 1 {
    let cols = self.feature_subsets[t]
    let sub_x = slice_matrix_cols(x, cols)
    let pred = self.trees[t].predict(sub_x)
    for i = 0; i < n_samples; i = i + 1 {
      let cls_idx = label_index(pred.get(i), self.classes)
      votes[i][cls_idx] = votes[i][cls_idx] + 1
    }
  }
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let mut best = 0
    let mut best_count = -1
    for c = 0; c < self.classes.length(); c = c + 1 {
      if votes[i][c] > best_count {
        best_count = votes[i][c]
        best = c
      }
    }
    out[i] = self.classes[best]
  }
  @math.Vector::new(out)
}

///|
pub fn RandomForestClassifier::score(
  self : RandomForestClassifier,
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

///|
pub fn RandomForestClassifier::feature_importances(
  self : RandomForestClassifier,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  if self.trees.length() == 0 {
    return @math.Vector::new([])
  }
  let importances = Array::make(self.n_features, 0.0)
  for t = 0; t < self.trees.length(); t = t + 1 {
    let cols = self.feature_subsets[t]
    let tree_imp = self.trees[t].get_feature_importances()
    for j = 0; j < cols.length(); j = j + 1 {
      importances[cols[j]] = importances[cols[j]] + tree_imp.get(j)
    }
  }
  for j = 0; j < importances.length(); j = j + 1 {
    importances[j] = importances[j] / self.trees.length().to_double()
  }
  @math.Vector::new(importances)
}

///|
fn assert_binary_labels_tree(y : @math.Vector) -> Unit {
  for i = 0; i < y.size(); i = i + 1 {
    let v = y.get(i)
    if not(v == 0.0 || v == 1.0) {
      abort("仅支持 {0,1} 标签")
    }
  }
}

///|
fn to_pm1(v : Double) -> Double {
  if v == 0.0 {
    -1.0
  } else if v == 1.0 {
    1.0
  } else {
    abort("标签必须是 0 或 1")
  }
}

///|
fn normalize_weights(weights : Array[Double]) -> Unit {
  let mut total = 0.0
  for i = 0; i < weights.length(); i = i + 1 {
    total = total + weights[i]
  }
  if total == 0.0 {
    return
  }
  for i = 0; i < weights.length(); i = i + 1 {
    weights[i] = weights[i] / total
  }
}

///|
fn sigmoid_tree(z : Double) -> Double {
  1.0 / (1.0 + @moonbitlang/core/math.exp(-z))
}

///|
/// ExtraTrees 分类器（极端随机树）
pub struct ExtraTreesClassifier {
  n_estimators : Int
  max_depth : Int
  min_samples_split : Int
  min_samples_leaf : Int
  max_features : Int
  seed : Int
  mut trees : Array[DecisionTreeClassifier]
  mut feature_subsets : Array[Array[Int]]
  mut classes : Array[Double]
  mut n_features : Int
  mut is_fitted : Bool
}

///|
pub fn ExtraTreesClassifier::new(
  n_estimators? : Int = 10,
  max_depth? : Int = 5,
  min_samples_split? : Int = 2,
  min_samples_leaf? : Int = 1,
  max_features? : Int = 0,
  seed? : Int = 7,
) -> ExtraTreesClassifier {
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
    classes: [],
    n_features: 0,
    is_fitted: false,
  }
}

///|
pub fn ExtraTreesClassifier::fit(
  self : ExtraTreesClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  self.classes = unique_labels(y)
  self.n_features = n_features
  self.trees = Array::make(self.n_estimators, DecisionTreeClassifier::new())
  self.feature_subsets = Array::make(self.n_estimators, [])
  let mut seed = self.seed
  for t = 0; t < self.n_estimators; t = t + 1 {
    seed = lcg_next(seed)
    let cols = select_feature_subset(n_features, self.max_features, seed)
    self.feature_subsets[t] = cols
    let sub_x = slice_matrix_cols(x, cols)
    let tree = DecisionTreeClassifier::new(
      max_depth=self.max_depth,
      min_samples_split=self.min_samples_split,
      min_samples_leaf=self.min_samples_leaf,
      random_splits=true,
      seed~,
    )
    tree.fit(sub_x, y)
    self.trees[t] = tree
  }
  self.is_fitted = true
}

///|
pub fn ExtraTreesClassifier::predict(
  self : ExtraTreesClassifier,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let votes = Array::make(n_samples, Array::make(self.classes.length(), 0))
  for t = 0; t < self.trees.length(); t = t + 1 {
    let cols = self.feature_subsets[t]
    let sub_x = slice_matrix_cols(x, cols)
    let pred = self.trees[t].predict(sub_x)
    for i = 0; i < n_samples; i = i + 1 {
      let cls_idx = label_index(pred.get(i), self.classes)
      votes[i][cls_idx] = votes[i][cls_idx] + 1
    }
  }
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let mut best = 0
    let mut best_count = -1
    for c = 0; c < self.classes.length(); c = c + 1 {
      if votes[i][c] > best_count {
        best_count = votes[i][c]
        best = c
      }
    }
    out[i] = self.classes[best]
  }
  @math.Vector::new(out)
}

///|
pub fn ExtraTreesClassifier::feature_importances(
  self : ExtraTreesClassifier,
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

///|
/// AdaBoost 二分类（使用决策树桩）
pub struct AdaBoostClassifier {
  n_estimators : Int
  learning_rate : Double
  max_depth : Int
  seed : Int
  mut estimators : Array[DecisionTreeClassifier]
  mut estimator_weights : Array[Double]
  mut classes : Array[Double]
  mut is_fitted : Bool
}

///|
pub fn AdaBoostClassifier::new(
  n_estimators? : Int = 20,
  learning_rate? : Double = 1.0,
  max_depth? : Int = 1,
  seed? : Int = 11,
) -> AdaBoostClassifier {
  if n_estimators <= 0 {
    abort("n_estimators 必须大于 0")
  }
  if learning_rate <= 0.0 {
    abort("learning_rate 必须大于 0")
  }
  {
    n_estimators,
    learning_rate,
    max_depth,
    seed,
    estimators: [],
    estimator_weights: [],
    classes: [],
    is_fitted: false,
  }
}

///|
pub fn AdaBoostClassifier::fit(
  self : AdaBoostClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  assert_binary_labels_tree(y)
  self.classes = unique_labels(y)
  let weights = Array::make(n_samples, 1.0 / n_samples.to_double())
  self.estimators = Array::make(
    self.n_estimators,
    DecisionTreeClassifier::new(),
  )
  self.estimator_weights = Array::make(self.n_estimators, 0.0)
  let mut seed = self.seed
  for m = 0; m < self.n_estimators; m = m + 1 {
    // 按权重有放回抽样
    let indices = Array::make(n_samples, 0)
    let cumsum = Array::make(n_samples, 0.0)
    let mut acc = 0.0
    for i = 0; i < n_samples; i = i + 1 {
      acc = acc + weights[i]
      cumsum[i] = acc
    }
    seed = lcg_next(seed)
    for i = 0; i < n_samples; i = i + 1 {
      seed = lcg_next(seed)
      let r = seed.to_double() / (0x7fffffff).to_double() // 伪随机 0-1
      let mut chosen = n_samples - 1
      for j = 0; j < n_samples; j = j + 1 {
        if r <= cumsum[j] {
          chosen = j
          break
        }
      }
      indices[i] = chosen
    }
    let boot_x = @math.Matrix::zeros(n_samples, n_features)
    let boot_y = Array::make(n_samples, 0.0)
    for i = 0; i < n_samples; i = i + 1 {
      let idx = indices[i]
      let row = x.get_row(idx)
      for f = 0; f < row.size(); f = f + 1 {
        boot_x.set(i, f, row.get(f))
      }
      boot_y[i] = y.get(idx)
    }
    let tree = DecisionTreeClassifier::new(
      max_depth=self.max_depth,
      min_samples_split=2,
      min_samples_leaf=1,
      seed~,
    )
    tree.fit(boot_x, @math.Vector::new(boot_y))
    let pred = tree.predict(x)
    let mut err = 0.0
    for i = 0; i < n_samples; i = i + 1 {
      let yi = to_pm1(y.get(i))
      let pi = to_pm1(pred.get(i))
      if yi != pi {
        err = err + weights[i]
      }
    }
    if err >= 0.5 {
      continue
    }
    let eps = 1.0e-12
    let alpha = 0.5 *
      @moonbitlang/core/math.ln((1.0 - err + eps) / (err + eps)) *
      self.learning_rate
    self.estimators[m] = tree
    self.estimator_weights[m] = alpha
    for i = 0; i < n_samples; i = i + 1 {
      let yi = to_pm1(y.get(i))
      let pi = to_pm1(pred.get(i))
      weights[i] = weights[i] * @moonbitlang/core/math.exp(-alpha * yi * pi)
    }
    normalize_weights(weights)
  }
  self.is_fitted = true
}

///|
pub fn AdaBoostClassifier::predict(
  self : AdaBoostClassifier,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let scores = Array::make(n_samples, 0.0)
  for m = 0; m < self.estimators.length(); m = m + 1 {
    let alpha = self.estimator_weights[m]
    if alpha == 0.0 || not(self.estimators[m].is_fitted) {
      continue
    }
    let pred = self.estimators[m].predict(x)
    for i = 0; i < n_samples; i = i + 1 {
      scores[i] = scores[i] + alpha * to_pm1(pred.get(i))
    }
  }
  let out = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    out[i] = if scores[i] >= 0.0 { 1.0 } else { 0.0 }
  }
  @math.Vector::new(out)
}

///|
/// 梯度提升二分类（对数损失）
pub struct GradientBoostingClassifier {
  n_estimators : Int
  learning_rate : Double
  max_depth : Int
  mut trees : Array[DecisionTreeRegressor]
  mut init_log_odds : Double
  mut is_fitted : Bool
}

///|
pub fn GradientBoostingClassifier::new(
  n_estimators? : Int = 50,
  learning_rate? : Double = 0.1,
  max_depth? : Int = 2,
) -> GradientBoostingClassifier {
  if n_estimators <= 0 {
    abort("n_estimators 必须大于 0")
  }
  if learning_rate <= 0.0 {
    abort("learning_rate 必须大于 0")
  }
  {
    n_estimators,
    learning_rate,
    max_depth,
    trees: [],
    init_log_odds: 0.0,
    is_fitted: false,
  }
}

///|
pub fn GradientBoostingClassifier::fit(
  self : GradientBoostingClassifier,
  x : @math.Matrix,
  y : @math.Vector,
) -> Unit {
  let (n_samples, _) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  assert_binary_labels_tree(y)
  let mut pos = 0.0
  for i = 0; i < y.size(); i = i + 1 {
    pos = pos + y.get(i)
  }
  let mut p = pos / n_samples.to_double()
  if p < 1.0e-6 {
    p = 1.0e-6
  }
  if p > 1.0 - 1.0e-6 {
    p = 1.0 - 1.0e-6
  }
  self.init_log_odds = @moonbitlang/core/math.ln(p / (1.0 - p))
  let scores = Array::make(n_samples, self.init_log_odds)
  self.trees = Array::make(self.n_estimators, DecisionTreeRegressor::new())
  for m = 0; m < self.n_estimators; m = m + 1 {
    let residuals = Array::make(n_samples, 0.0)
    for i = 0; i < n_samples; i = i + 1 {
      let prob = sigmoid_tree(scores[i])
      residuals[i] = y.get(i) - prob
    }
    let tree = DecisionTreeRegressor::new(max_depth=self.max_depth)
    tree.fit(x, @math.Vector::new(residuals))
    let update = tree.predict(x)
    for i = 0; i < n_samples; i = i + 1 {
      scores[i] = scores[i] + self.learning_rate * update.get(i)
    }
    self.trees[m] = tree
  }
  self.is_fitted = true
}

///|
pub fn GradientBoostingClassifier::predict_proba(
  self : GradientBoostingClassifier,
  x : @math.Matrix,
) -> @math.Vector {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let scores = Array::make(n_samples, self.init_log_odds)
  for m = 0; m < self.trees.length(); m = m + 1 {
    let update = self.trees[m].predict(x)
    for i = 0; i < n_samples; i = i + 1 {
      scores[i] = scores[i] + self.learning_rate * update.get(i)
    }
  }
  for i = 0; i < n_samples; i = i + 1 {
    scores[i] = sigmoid_tree(scores[i])
  }
  @math.Vector::new(scores)
}

///|
pub fn GradientBoostingClassifier::predict(
  self : GradientBoostingClassifier,
  x : @math.Matrix,
) -> @math.Vector {
  let prob = self.predict_proba(x)
  let out = Array::make(prob.size(), 0.0)
  for i = 0; i < prob.size(); i = i + 1 {
    out[i] = if prob.get(i) >= 0.5 { 1.0 } else { 0.0 }
  }
  @math.Vector::new(out)
}
