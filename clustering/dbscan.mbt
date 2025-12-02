///|
/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
/// 基于密度的聚类算法

///|
/// DBSCAN 聚类器
pub struct DBSCAN {
  eps : Double // 邻域半径
  min_samples : Int // 最小样本数
  mut labels : Array[Int] // 样本标签 (-1 表示噪声)
  mut is_fitted : Bool
}

///|
pub fn DBSCAN::new(eps : Double, min_samples : Int) -> DBSCAN {
  if eps <= 0.0 {
    abort("eps 必须大于 0")
  }
  if min_samples <= 0 {
    abort("min_samples 必须大于 0")
  }
  { eps, min_samples, labels: [], is_fitted: false }
}

///|
/// 训练 DBSCAN 模型
pub fn DBSCAN::fit(self : DBSCAN, x : @math.Matrix) -> Unit {
  let (n_samples, _) = x.shape()
  if n_samples == 0 {
    abort("数据不能为空")
  }
  self.labels = Array::make(n_samples, -2) // -2 表示未访问
  let mut cluster_id = 0
  for i = 0; i < n_samples; i = i + 1 {
    if self.labels[i] != -2 {
      continue
    }

    // 找到邻居
    let neighbors = find_neighbors(x, i, self.eps)
    if neighbors.length() < self.min_samples {
      // 标记为噪声
      self.labels[i] = -1
    } else {
      // 扩展簇
      expand_cluster(
        x,
        i,
        neighbors,
        cluster_id,
        self.labels,
        self.eps,
        self.min_samples,
      )
      cluster_id = cluster_id + 1
    }
  }
  self.is_fitted = true
}

///|
/// 训练并返回标签
pub fn DBSCAN::fit_predict(self : DBSCAN, x : @math.Matrix) -> Array[Int] {
  self.fit(x)
  self.labels
}

///|
/// 获取标签
pub fn DBSCAN::get_labels(self : DBSCAN) -> Array[Int] {
  self.labels
}

///|
/// 获取簇的数量（不包括噪声）
pub fn DBSCAN::get_n_clusters(self : DBSCAN) -> Int {
  let mut max_label = -1
  for i = 0; i < self.labels.length(); i = i + 1 {
    if self.labels[i] > max_label {
      max_label = self.labels[i]
    }
  }
  max_label + 1
}

///|
/// ========== 辅助函数 ==========

///|
/// 找到一个点的所有邻居
fn find_neighbors(
  x : @math.Matrix,
  point_idx : Int,
  eps : Double,
) -> Array[Int] {
  let (n_samples, _) = x.shape()
  let neighbors = []
  let point = x.get_row(point_idx)
  for i = 0; i < n_samples; i = i + 1 {
    let other = x.get_row(i)
    let dist = euclidean_distance_dbscan(point, other)
    if dist <= eps {
      neighbors.push(i)
    }
  }
  neighbors
}

///|
/// 扩展簇
fn expand_cluster(
  x : @math.Matrix,
  point_idx : Int,
  neighbors : Array[Int],
  cluster_id : Int,
  labels : Array[Int],
  eps : Double,
  min_samples : Int,
) -> Unit {
  labels[point_idx] = cluster_id
  let mut i = 0
  while i < neighbors.length() {
    let neighbor_idx = neighbors[i]
    if labels[neighbor_idx] == -1 {
      // 噪声点变成边界点
      labels[neighbor_idx] = cluster_id
    } else if labels[neighbor_idx] == -2 {
      // 未访问的点
      labels[neighbor_idx] = cluster_id

      // 找到这个点的邻居
      let new_neighbors = find_neighbors(x, neighbor_idx, eps)
      if new_neighbors.length() >= min_samples {
        // 核心点，添加其邻居
        for j = 0; j < new_neighbors.length(); j = j + 1 {
          let new_neighbor = new_neighbors[j]
          let mut already_in = false
          for k = 0; k < neighbors.length(); k = k + 1 {
            if neighbors[k] == new_neighbor {
              already_in = true
              break
            }
          }
          if not(already_in) {
            neighbors.push(new_neighbor)
          }
        }
      }
    }
    i = i + 1
  }
}

///|
/// 欧几里得距离
fn euclidean_distance_dbscan(a : @math.Vector, b : @math.Vector) -> Double {
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
