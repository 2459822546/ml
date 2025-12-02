///|
/// 特征选择工具
/// 提供方差过滤、相关性过滤、互信息和线性回归版 RFE

///|
/// 从矩阵取出某列为新数组
fn column_to_array(matrix : @math.Matrix, col : Int) -> Array[Double] {
  let (rows, _) = matrix.shape()
  let col_vec = matrix.get_col(col)
  let arr = Array::make(rows, 0.0)
  for i = 0; i < rows; i = i + 1 {
    arr[i] = col_vec.get(i)
  }
  arr
}

///|
/// 方差过滤：返回保留的列索引
pub fn variance_threshold(x : @math.Matrix, threshold : Double) -> Array[Int] {
  let (_, n_features) = x.shape()
  if threshold < 0.0 {
    abort("threshold 不能为负")
  }
  let kept = Array::make(0, 0)
  for j = 0; j < n_features; j = j + 1 {
    let col = column_to_array(x, j)
    if variance_array(col) > threshold {
      kept.push(j)
    }
  }
  kept
}

///|
/// 相关性过滤：按顺序保留与已选特征的绝对相关性低于阈值的列
pub fn correlation_filter(x : @math.Matrix, threshold : Double) -> Array[Int] {
  let (_, n_features) = x.shape()
  if threshold <= 0.0 || threshold >= 1.0 {
    abort("threshold 必须在 (0, 1) 之间")
  }
  let kept = Array::make(0, 0)
  for j = 0; j < n_features; j = j + 1 {
    let col_j = column_to_array(x, j)
    let mut correlated = false
    for k_idx = 0; k_idx < kept.length(); k_idx = k_idx + 1 {
      let k = kept[k_idx]
      let col_k = column_to_array(x, k)
      let corr = @math.correlation(col_j, col_k).abs()
      if corr >= threshold {
        correlated = true
        break
      }
    }
    if not(correlated) {
      kept.push(j)
    }
  }
  kept
}

///|
/// 将连续值按等宽分桶，返回桶索引
fn bin_indices(values : Array[Double], n_bins : Int) -> Array[Int] {
  if n_bins <= 1 {
    abort("n_bins 必须大于 1")
  }
  let n = values.length()
  if n == 0 {
    return []
  }
  let mut min_v = values[0]
  let mut max_v = values[0]
  for i = 1; i < n; i = i + 1 {
    if values[i] < min_v {
      min_v = values[i]
    }
    if values[i] > max_v {
      max_v = values[i]
    }
  }
  let range = max_v - min_v
  if range == 0.0 {
    return Array::make(n, 0)
  }
  let result = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    let ratio = (values[i] - min_v) / range
    let idx = (ratio * n_bins.to_double()).to_int()
    result[i] = if idx >= n_bins { n_bins - 1 } else { idx }
  }
  result
}

///|
/// 单列与标签的互信息（基于等宽分桶）
fn mutual_information_single(
  feature : Array[Double],
  labels : Array[Double],
  n_bins : Int,
) -> Double {
  if feature.length() != labels.length() {
    abort("特征与标签长度不一致")
  }
  let n = feature.length()
  if n == 0 {
    abort("空数据无法计算互信息")
  }
  let bins_x = bin_indices(feature, n_bins)
  let bins_y = bin_indices(labels, n_bins)

  // 统计联合与边缘分布
  let joint = Array::make(n_bins, [])
  for i = 0; i < n_bins; i = i + 1 {
    joint[i] = Array::make(n_bins, 0.0)
  }
  let px = Array::make(n_bins, 0.0)
  let py = Array::make(n_bins, 0.0)
  for i = 0; i < n; i = i + 1 {
    let bx = bins_x[i]
    let by = bins_y[i]
    joint[bx][by] = joint[bx][by] + 1.0
    px[bx] = px[bx] + 1.0
    py[by] = py[by] + 1.0
  }

  // 互信息计算，使用自然对数
  let n_double = n.to_double()
  let mut mi = 0.0
  for i = 0; i < n_bins; i = i + 1 {
    for j = 0; j < n_bins; j = j + 1 {
      let count = joint[i][j]
      if count == 0.0 {
        continue
      }
      let pxy = count / n_double
      let px_i = px[i] / n_double
      let py_j = py[j] / n_double
      mi = mi + pxy * @moonbitlang/core/math.ln(pxy / (px_i * py_j))
    }
  }
  mi
}

///|
/// 逐列互信息分数
pub fn mutual_information_scores(
  x : @math.Matrix,
  y : @math.Vector,
  n_bins? : Int = 10,
) -> Array[Double] {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  let scores = Array::make(n_features, 0.0)
  for j = 0; j < n_features; j = j + 1 {
    let col = column_to_array(x, j)
    scores[j] = mutual_information_single(col, y.to_array(), n_bins)
  }
  scores
}

///|
/// 根据互信息选择前 k 个特征，返回列索引（降序得分）
pub fn select_by_mutual_information(
  x : @math.Matrix,
  y : @math.Vector,
  k : Int,
  n_bins? : Int = 10,
) -> Array[Int] {
  let (_, n_features) = x.shape()
  if k <= 0 || k > n_features {
    abort("k 必须在 (0, n_features] 之间")
  }
  let scores = mutual_information_scores(x, y, n_bins~)
  // 简单排序索引
  let indices = Array::make(n_features, 0)
  for i = 0; i < n_features; i = i + 1 {
    indices[i] = i
  }
  indices.sort_by(fn(a, b) {
    // 降序排序
    let diff = scores[b] - scores[a]
    if diff > 0.0 {
      1
    } else if diff < 0.0 {
      -1
    } else {
      0
    }
  })
  let selected = Array::make(k, 0)
  for i = 0; i < k; i = i + 1 {
    selected[i] = indices[i]
  }
  selected
}

///|
/// 递归特征消除（线性回归版）
pub fn rfe_linear_regression(
  x : @math.Matrix,
  y : @math.Vector,
  n_features_to_select : Int,
) -> Array[Int] {
  let (n_samples, n_features) = x.shape()
  if y.size() != n_samples {
    abort("X 和 y 的样本数不匹配")
  }
  if n_features_to_select <= 0 || n_features_to_select > n_features {
    abort("n_features_to_select 超出范围")
  }

  // 当前特征索引列表
  let mut active = Array::make(n_features, 0)
  for j = 0; j < n_features; j = j + 1 {
    active[j] = j
  }
  while active.length() > n_features_to_select {
    // 构建子矩阵
    let sub_cols = active.length()
    let sub_x = @math.Matrix::zeros(n_samples, sub_cols)
    for i = 0; i < n_samples; i = i + 1 {
      let row = x.get_row(i)
      for j_idx = 0; j_idx < sub_cols; j_idx = j_idx + 1 {
        sub_x.set(i, j_idx, row.get(active[j_idx]))
      }
    }

    // 训练线性回归并找到绝对权重最小的列
    let model = @linear.LinearRegression::new()
    model.fit(sub_x, y)
    let weights = model.get_weights()
    let mut min_idx = 0
    let mut min_abs = weights.get(0).abs()
    for j = 1; j < weights.size(); j = j + 1 {
      let abs_w = weights.get(j).abs()
      if abs_w < min_abs {
        min_abs = abs_w
        min_idx = j
      }
    }
    let filtered = Array::make(active.length() - 1, 0)
    let mut t = 0
    for j = 0; j < active.length(); j = j + 1 {
      if j != min_idx {
        filtered[t] = active[j]
        t = t + 1
      }
    }
    active = filtered
  }

  // 返回升序索引便于使用
  active.sort_by(fn(a, b) { a - b })
  active
}
