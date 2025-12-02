///|
/// K-Means 聚类算法
/// 包括标准 K-Means 和 K-Means++

///|
/// K-Means 聚类器
pub struct KMeans {
  n_clusters : Int // 簇的数量
  max_iter : Int // 最大迭代次数
  tol : Double // 收敛阈值
  init_method : InitMethod // 初始化方法
  mut centers : @math.Matrix // 簇中心
  mut labels : Array[Int] // 样本标签
  mut inertia : Double // 簇内平方和
  mut is_fitted : Bool
}

///|
/// 初始化方法
pub(all) enum InitMethod {
  Random // 随机初始化
  KMeansPlusPlus // K-Means++ 初始化
}

///|
pub fn KMeans::new(
  n_clusters : Int,
  max_iter : Int,
  tol : Double,
  init_method : InitMethod,
) -> KMeans {
  if n_clusters <= 0 {
    abort("n_clusters 必须大于 0")
  }
  if max_iter <= 0 {
    abort("max_iter 必须大于 0")
  }
  if tol <= 0.0 {
    abort("tol 必须大于 0")
  }
  {
    n_clusters,
    max_iter,
    tol,
    init_method,
    centers: @math.Matrix::new([[0.0]]), // 占位，fit 时会覆盖
    labels: [],
    inertia: 0.0,
    is_fitted: false,
  }
}

///|
/// 训练 K-Means 模型
pub fn KMeans::fit(self : KMeans, x : @math.Matrix) -> Unit {
  let (n_samples, n_features) = x.shape()
  if n_samples < self.n_clusters {
    abort("样本数量必须大于簇的数量")
  }

  // 初始化簇中心
  self.centers = match self.init_method {
    Random => init_centers_random(x, self.n_clusters)
    KMeansPlusPlus => init_centers_kmeans_plusplus(x, self.n_clusters)
  }
  self.labels = Array::make(n_samples, 0)

  // K-Means 迭代
  for iter = 0; iter < self.max_iter; iter = iter + 1 {
    // 分配样本到最近的簇
    let mut changed = false
    for i = 0; i < n_samples; i = i + 1 {
      let sample = x.get_row(i)
      let new_label = find_nearest_center(sample, self.centers)
      if self.labels[i] != new_label {
        changed = true
        self.labels[i] = new_label
      }
    }

    // 如果没有变化，提前停止
    if not(changed) && iter > 0 {
      break
    }

    // 更新簇中心
    let old_centers = self.centers.copy()
    self.centers = update_centers(x, self.labels, self.n_clusters, n_features)

    // 检查收敛
    let center_shift = compute_center_shift(old_centers, self.centers)
    if center_shift < self.tol {
      break
    }
  }

  // 计算惯性（簇内平方和）
  self.inertia = compute_inertia(x, self.centers, self.labels)
  self.is_fitted = true
}

///|
/// 预测样本所属的簇
pub fn KMeans::predict(self : KMeans, x : @math.Matrix) -> Array[Int] {
  if not(self.is_fitted) {
    abort("模型尚未训练")
  }
  let (n_samples, _) = x.shape()
  let labels = Array::make(n_samples, 0)
  for i = 0; i < n_samples; i = i + 1 {
    let sample = x.get_row(i)
    labels[i] = find_nearest_center(sample, self.centers)
  }
  labels
}

///|
/// 训练并预测
pub fn KMeans::fit_predict(self : KMeans, x : @math.Matrix) -> Array[Int] {
  self.fit(x)
  self.labels
}

///|
/// 获取簇中心
pub fn KMeans::get_centers(self : KMeans) -> @math.Matrix {
  self.centers
}

///|
/// 获取惯性
pub fn KMeans::get_inertia(self : KMeans) -> Double {
  self.inertia
}

///|
/// ========== 辅助函数 ==========

///|
/// 随机初始化簇中心
fn init_centers_random(x : @math.Matrix, n_clusters : Int) -> @math.Matrix {
  let (n_samples, n_features) = x.shape()
  let centers_data = Array::make(n_clusters, [])

  // 简单的随机选择（不重复）
  let selected = []
  for cluster_idx = 0; cluster_idx < n_clusters; cluster_idx = cluster_idx + 1 {
    let mut idx = 0
    let mut found = false
    for attempt = 0; attempt < 100; attempt = attempt + 1 {
      idx = (attempt * 7 + 13) % n_samples // 简单的伪随机
      let mut already_selected = false
      for j = 0; j < selected.length(); j = j + 1 {
        if selected[j] == idx {
          already_selected = true
          break
        }
      }
      if not(already_selected) {
        found = true
        break
      }
    }
    if not(found) {
      idx = selected.length() % n_samples
    }
    selected.push(idx)
  }
  for i = 0; i < n_clusters; i = i + 1 {
    let row = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      row[j] = x.get(selected[i], j)
    }
    centers_data[i] = row
  }
  @math.Matrix::new(centers_data)
}

///|
/// K-Means++ 初始化
fn init_centers_kmeans_plusplus(
  x : @math.Matrix,
  n_clusters : Int,
) -> @math.Matrix {
  let (n_samples, n_features) = x.shape()
  let centers_data = Array::make(n_clusters, [])

  // 第一个中心：随机选择
  let first_idx = 0 // 简化：选择第一个样本
  let first_center = Array::make(n_features, 0.0)
  for j = 0; j < n_features; j = j + 1 {
    first_center[j] = x.get(first_idx, j)
  }
  centers_data[0] = first_center

  // 选择剩余的中心
  for k = 1; k < n_clusters; k = k + 1 {
    // 计算每个样本到最近中心的距离平方
    let distances = Array::make(n_samples, 0.0)
    for i = 0; i < n_samples; i = i + 1 {
      let sample = x.get_row(i)
      let mut min_dist = @double.infinity
      for c = 0; c < k; c = c + 1 {
        let center = @math.Vector::new(centers_data[c])
        let dist = euclidean_distance_kmeans(sample, center)
        if dist < min_dist {
          min_dist = dist
        }
      }
      distances[i] = min_dist * min_dist
    }

    // 选择距离最大的样本作为下一个中心
    let mut max_dist = 0.0
    let mut max_idx = 0
    for i = 0; i < n_samples; i = i + 1 {
      if distances[i] > max_dist {
        max_dist = distances[i]
        max_idx = i
      }
    }
    let new_center = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      new_center[j] = x.get(max_idx, j)
    }
    centers_data[k] = new_center
  }
  @math.Matrix::new(centers_data)
}

///|
/// 找到最近的簇中心
fn find_nearest_center(sample : @math.Vector, centers : @math.Matrix) -> Int {
  let (n_centers, _) = centers.shape()
  let mut min_dist = @double.infinity
  let mut nearest = 0
  for i = 0; i < n_centers; i = i + 1 {
    let center = centers.get_row(i)
    let dist = euclidean_distance_kmeans(sample, center)
    if dist < min_dist {
      min_dist = dist
      nearest = i
    }
  }
  nearest
}

///|
/// 更新簇中心
fn update_centers(
  x : @math.Matrix,
  labels : Array[Int],
  n_clusters : Int,
  n_features : Int,
) -> @math.Matrix {
  let centers_data = Array::make(n_clusters, [])
  let counts = Array::make(n_clusters, 0)

  // 初始化
  for i = 0; i < n_clusters; i = i + 1 {
    centers_data[i] = Array::make(n_features, 0.0)
  }

  // 累加每个簇的样本
  let (n_samples, _) = x.shape()
  for i = 0; i < n_samples; i = i + 1 {
    let label = labels[i]
    counts[label] = counts[label] + 1
    for j = 0; j < n_features; j = j + 1 {
      centers_data[label][j] = centers_data[label][j] + x.get(i, j)
    }
  }

  // 计算平均值
  for i = 0; i < n_clusters; i = i + 1 {
    if counts[i] > 0 {
      for j = 0; j < n_features; j = j + 1 {
        centers_data[i][j] = centers_data[i][j] / counts[i].to_double()
      }
    }
  }
  @math.Matrix::new(centers_data)
}

///|
/// 计算簇中心的偏移量
fn compute_center_shift(
  old_centers : @math.Matrix,
  new_centers : @math.Matrix,
) -> Double {
  let (n_centers, _) = old_centers.shape()
  let mut total_shift = 0.0
  for i = 0; i < n_centers; i = i + 1 {
    let old_center = old_centers.get_row(i)
    let new_center = new_centers.get_row(i)
    let shift = euclidean_distance_kmeans(old_center, new_center)
    total_shift = total_shift + shift
  }
  total_shift / n_centers.to_double()
}

///|
/// 计算惯性（簇内平方和）
fn compute_inertia(
  x : @math.Matrix,
  centers : @math.Matrix,
  labels : Array[Int],
) -> Double {
  let (n_samples, _) = x.shape()
  let mut inertia = 0.0
  for i = 0; i < n_samples; i = i + 1 {
    let sample = x.get_row(i)
    let center = centers.get_row(labels[i])
    let dist = euclidean_distance_kmeans(sample, center)
    inertia = inertia + dist * dist
  }
  inertia
}

///|
/// 欧几里得距离
fn euclidean_distance_kmeans(a : @math.Vector, b : @math.Vector) -> Double {
  if a.size() != b.size() {
    abort("向量维度不匹配")
  }
  let mut sum = 0.0
  for i = 0; i < a.size(); i = i + 1 {
    let diff = a.get(i) - b.get(i)
    sum = sum + diff * diff
  }
  sum.sqrt()
}
