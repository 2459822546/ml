///|
/// 层次聚类 (Hierarchical Clustering)
/// 支持凝聚层次聚类

///|
/// 层次聚类器
pub struct AgglomerativeClustering {
  n_clusters : Int // 簇的数量
  linkage : LinkageMethod // 链接方法
  mut labels : Array[Int] // 样本标签
  mut is_fitted : Bool
}

///|
/// 链接方法
pub(all) enum LinkageMethod {
  Single // 单链接（最小距离）
  Complete // 全链接（最大距离）
  Average // 平均链接
}

///|
pub fn AgglomerativeClustering::new(
  n_clusters : Int,
  linkage : LinkageMethod,
) -> AgglomerativeClustering {
  if n_clusters <= 0 {
    abort("n_clusters 必须大于 0")
  }
  { n_clusters, linkage, labels: [], is_fitted: false }
}

///|
/// 训练层次聚类模型
pub fn AgglomerativeClustering::fit(
  self : AgglomerativeClustering,
  x : @math.Matrix,
) -> Unit {
  let (n_samples, _) = x.shape()
  if n_samples == 0 {
    abort("数据不能为空")
  }
  if n_samples < self.n_clusters {
    abort("样本数量必须大于簇的数量")
  }

  // 初始化：每个样本是一个簇
  let mut clusters = []
  for i = 0; i < n_samples; i = i + 1 {
    clusters.push([i])
  }

  // 计算距离矩阵
  let distance_matrix = compute_distance_matrix(x)

  // 凝聚合并
  while clusters.length() > self.n_clusters {
    // 找到最近的两个簇
    let (i, j) = find_closest_clusters(clusters, distance_matrix, self.linkage)

    // 合并簇
    let merged = []
    for k = 0; k < clusters[i].length(); k = k + 1 {
      merged.push(clusters[i][k])
    }
    for k = 0; k < clusters[j].length(); k = k + 1 {
      merged.push(clusters[j][k])
    }

    // 移除旧簇并添加新簇
    let new_clusters = []
    for k = 0; k < clusters.length(); k = k + 1 {
      if k != i && k != j {
        new_clusters.push(clusters[k])
      }
    }
    new_clusters.push(merged)
    clusters = new_clusters
  }

  // 分配标签
  self.labels = Array::make(n_samples, 0)
  for cluster_id = 0
      cluster_id < clusters.length()
      cluster_id = cluster_id + 1 {
    for i = 0; i < clusters[cluster_id].length(); i = i + 1 {
      let sample_idx = clusters[cluster_id][i]
      self.labels[sample_idx] = cluster_id
    }
  }
  self.is_fitted = true
}

///|
/// 训练并返回标签
pub fn AgglomerativeClustering::fit_predict(
  self : AgglomerativeClustering,
  x : @math.Matrix,
) -> Array[Int] {
  self.fit(x)
  self.labels
}

///|
/// 获取标签
pub fn AgglomerativeClustering::get_labels(
  self : AgglomerativeClustering,
) -> Array[Int] {
  self.labels
}

///|
/// ========== 辅助函数 ==========

///|
/// 计算样本间的距离矩阵
fn compute_distance_matrix(x : @math.Matrix) -> @math.Matrix {
  let (n_samples, _) = x.shape()
  let dist_data = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(n_samples, 0.0)
    let sample_i = x.get_row(i)
    for j = 0; j < n_samples; j = j + 1 {
      if i == j {
        row[j] = 0.0
      } else {
        let sample_j = x.get_row(j)
        row[j] = euclidean_distance_hier(sample_i, sample_j)
      }
    }
    dist_data[i] = row
  }
  @math.Matrix::new(dist_data)
}

///|
/// 找到最近的两个簇
fn find_closest_clusters(
  clusters : Array[Array[Int]],
  distance_matrix : @math.Matrix,
  linkage : LinkageMethod,
) -> (Int, Int) {
  let mut min_dist = @double.infinity
  let mut best_i = 0
  let mut best_j = 1
  for i = 0; i < clusters.length(); i = i + 1 {
    for j = i + 1; j < clusters.length(); j = j + 1 {
      let dist = compute_cluster_distance(
        clusters[i],
        clusters[j],
        distance_matrix,
        linkage,
      )
      if dist < min_dist {
        min_dist = dist
        best_i = i
        best_j = j
      }
    }
  }
  (best_i, best_j)
}

///|
/// 计算两个簇之间的距离
fn compute_cluster_distance(
  cluster_a : Array[Int],
  cluster_b : Array[Int],
  distance_matrix : @math.Matrix,
  linkage : LinkageMethod,
) -> Double {
  match linkage {
    Single => {
      // 单链接：最小距离
      let mut min_dist = @double.infinity
      for i = 0; i < cluster_a.length(); i = i + 1 {
        for j = 0; j < cluster_b.length(); j = j + 1 {
          let dist = distance_matrix.get(cluster_a[i], cluster_b[j])
          if dist < min_dist {
            min_dist = dist
          }
        }
      }
      min_dist
    }
    Complete => {
      // 全链接：最大距离
      let mut max_dist = 0.0
      for i = 0; i < cluster_a.length(); i = i + 1 {
        for j = 0; j < cluster_b.length(); j = j + 1 {
          let dist = distance_matrix.get(cluster_a[i], cluster_b[j])
          if dist > max_dist {
            max_dist = dist
          }
        }
      }
      max_dist
    }
    Average => {
      // 平均链接：平均距离
      let mut sum_dist = 0.0
      for i = 0; i < cluster_a.length(); i = i + 1 {
        for j = 0; j < cluster_b.length(); j = j + 1 {
          sum_dist = sum_dist + distance_matrix.get(cluster_a[i], cluster_b[j])
        }
      }
      sum_dist / (cluster_a.length() * cluster_b.length()).to_double()
    }
  }
}

///|
/// 欧几里得距离
fn euclidean_distance_hier(a : @math.Vector, b : @math.Vector) -> Double {
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
