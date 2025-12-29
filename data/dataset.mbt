///|
/// Dataset - 数据集抽象
/// 用于封装特征和标签数据
pub struct Dataset {
  features : Array[Array[Double]] // 特征数据 X
  labels : Array[Double] // 标签数据 y
}

///|
/// 样本结构
pub struct Sample {
  features : Array[Double]
  label : Double
}

///|
/// 多标签样本
pub struct MultiSample {
  features : Array[Double]
  labels : Array[Double]
}

///|
/// 创建样本
pub fn Sample::new(features : Array[Double], label : Double) -> Sample {
  { features, label }
}

///|
/// 创建多标签样本
pub fn MultiSample::new(
  features : Array[Double],
  labels : Array[Double],
) -> MultiSample {
  { features, labels }
}

///|
/// 校验特征矩阵的维度，并确保标签数量一致
fn validate_features_and_labels(
  features : Array[Array[Double]],
  labels : Array[Double],
) -> Unit {
  if features.length() != labels.length() {
    abort("特征和标签的样本数不匹配")
  }
  if features.length() == 0 {
    return
  }
  let expected_dim = features[0].length()
  for i = 0; i < features.length(); i = i + 1 {
    if features[i].length() != expected_dim {
      abort("样本 \{i} 的特征维度不一致")
    }
  }
}

///|
/// 校验特征矩阵与多标签矩阵
fn validate_features_and_multilabels(
  features : Array[Array[Double]],
  labels : Array[Array[Double]],
) -> Unit {
  if features.length() != labels.length() {
    abort("特征和标签的样本数不匹配")
  }
  if features.length() == 0 {
    return
  }
  let expected_dim = features[0].length()
  let expected_labels = labels[0].length()
  for i = 0; i < features.length(); i = i + 1 {
    if features[i].length() != expected_dim {
      abort("样本 \{i} 的特征维度不一致")
    }
    if labels[i].length() != expected_labels {
      abort("样本 \{i} 的标签维度不一致")
    }
  }
}

///|
pub fn Dataset::new(
  features : Array[Array[Double]],
  labels : Array[Double],
) -> Dataset {
  validate_features_and_labels(features, labels)
  { features, labels }
}

///|
/// 从结构化样本创建 Dataset
pub fn Dataset::from_samples(samples : Array[Sample]) -> Dataset {
  let n_samples = samples.length()
  let features = Array::make(n_samples, [])
  let labels = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    features[i] = samples[i].features
    labels[i] = samples[i].label
  }
  Dataset::new(features, labels)
}

///|
/// 转换为结构化样本
pub fn Dataset::to_samples(self : Dataset) -> Array[Sample] {
  let n_samples = self.len()
  let samples = Array::make(n_samples, Sample::new([], 0.0))
  for i = 0; i < n_samples; i = i + 1 {
    samples[i] = Sample::new(self.features[i], self.labels[i])
  }
  samples
}

///|
/// 多标签数据集
pub struct MultiDataset {
  features : Array[Array[Double]]
  labels : Array[Array[Double]]
}

///|
pub fn MultiDataset::new(
  features : Array[Array[Double]],
  labels : Array[Array[Double]],
) -> MultiDataset {
  validate_features_and_multilabels(features, labels)
  { features, labels }
}

///|
/// 从多标签样本创建 MultiDataset
pub fn MultiDataset::from_samples(samples : Array[MultiSample]) -> MultiDataset {
  let n_samples = samples.length()
  let features = Array::make(n_samples, [])
  let labels = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    features[i] = samples[i].features
    labels[i] = samples[i].labels
  }
  MultiDataset::new(features, labels)
}

///|
/// 转换为结构化样本
pub fn MultiDataset::to_samples(self : MultiDataset) -> Array[MultiSample] {
  let n_samples = self.len()
  let samples = Array::make(n_samples, MultiSample::new([], []))
  for i = 0; i < n_samples; i = i + 1 {
    samples[i] = MultiSample::new(self.features[i], self.labels[i])
  }
  samples
}

///|
/// 获取样本数量
pub fn MultiDataset::len(self : MultiDataset) -> Int {
  self.features.length()
}

///|
/// 获取特征维度
pub fn MultiDataset::n_features(self : MultiDataset) -> Int {
  if self.len() == 0 {
    0
  } else {
    self.features[0].length()
  }
}

///|
/// 获取标签维度
pub fn MultiDataset::n_labels(self : MultiDataset) -> Int {
  if self.len() == 0 {
    0
  } else {
    self.labels[0].length()
  }
}

///|
/// 获取指定索引的样本
pub fn MultiDataset::get(
  self : MultiDataset,
  index : Int,
) -> (Array[Double], Array[Double]) {
  if index < 0 || index >= self.len() {
    abort("索引越界")
  }
  (self.features[index], self.labels[index])
}

///|
/// 批量获取样本
pub fn MultiDataset::get_batch(
  self : MultiDataset,
  indices : Array[Int],
) -> (Array[Array[Double]], Array[Array[Double]]) {
  let batch_size = indices.length()
  let batch_features = Array::make(batch_size, [])
  let batch_labels = Array::make(batch_size, [])
  for i = 0; i < batch_size; i = i + 1 {
    let idx = indices[i]
    if idx < 0 || idx >= self.len() {
      abort("索引越界")
    }
    batch_features[i] = self.features[idx]
    batch_labels[i] = self.labels[idx]
  }
  (batch_features, batch_labels)
}

///|
/// 获取所有特征
pub fn MultiDataset::get_features(self : MultiDataset) -> Array[Array[Double]] {
  self.features
}

///|
/// 获取所有标签
pub fn MultiDataset::get_labels(self : MultiDataset) -> Array[Array[Double]] {
  self.labels
}

///|
/// 获取指定范围的子集
pub fn MultiDataset::slice(
  self : MultiDataset,
  start : Int,
  end : Int,
) -> MultiDataset {
  if start < 0 || start >= self.len() {
    abort("起始索引越界")
  }
  if end < start || end > self.len() {
    abort("结束索引越界")
  }
  let n_samples = end - start
  let new_features = Array::make(n_samples, [])
  let new_labels = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    new_features[i] = self.features[start + i]
    new_labels[i] = self.labels[start + i]
  }
  MultiDataset::new(new_features, new_labels)
}

///|
/// 获取指定范围的子集（语义同 slice）
pub fn MultiDataset::range(
  self : MultiDataset,
  start : Int,
  end : Int,
) -> MultiDataset {
  self.slice(start, end)
}

///|
/// 映射数据集
pub fn MultiDataset::map(
  self : MultiDataset,
  mapper : (Array[Double], Array[Double]) -> (Array[Double], Array[Double]),
) -> MultiDataset {
  let n_samples = self.len()
  let new_features = Array::make(n_samples, [])
  let new_labels = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let (features, labels) = mapper(self.features[i], self.labels[i])
    new_features[i] = features
    new_labels[i] = labels
  }
  MultiDataset::new(new_features, new_labels)
}

///|
/// 打乱数据集
pub fn MultiDataset::shuffle(self : MultiDataset, seed : Int) -> MultiDataset {
  let n_samples = self.len()
  if n_samples <= 1 {
    return MultiDataset::new(self.features, self.labels)
  }
  let indices = Array::make(n_samples, 0)
  for i = 0; i < n_samples; i = i + 1 {
    indices[i] = i
  }
  shuffle_array(indices, seed)
  let new_features = Array::make(n_samples, [])
  let new_labels = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let idx = indices[i]
    new_features[i] = self.features[idx]
    new_labels[i] = self.labels[idx]
  }
  MultiDataset::new(new_features, new_labels)
}

///|
/// 获取样本数量
pub fn Dataset::len(self : Dataset) -> Int {
  self.features.length()
}

///|
/// 获取特征维度
pub fn Dataset::n_features(self : Dataset) -> Int {
  if self.len() == 0 {
    0
  } else {
    self.features[0].length()
  }
}

///|
/// 获取指定索引的样本
pub fn Dataset::get(self : Dataset, index : Int) -> (Array[Double], Double) {
  if index < 0 || index >= self.len() {
    abort("索引越界")
  }
  (self.features[index], self.labels[index])
}

///|
/// 批量获取样本
pub fn Dataset::get_batch(
  self : Dataset,
  indices : Array[Int],
) -> (Array[Array[Double]], Array[Double]) {
  let batch_size = indices.length()
  let batch_features = Array::make(batch_size, [])
  let batch_labels = Array::make(batch_size, 0.0)
  for i = 0; i < batch_size; i = i + 1 {
    let idx = indices[i]
    if idx < 0 || idx >= self.len() {
      abort("索引越界")
    }
    batch_features[i] = self.features[idx]
    batch_labels[i] = self.labels[idx]
  }
  (batch_features, batch_labels)
}

///|
/// 获取所有特征
pub fn Dataset::get_features(self : Dataset) -> Array[Array[Double]] {
  self.features
}

///|
/// 获取所有标签
pub fn Dataset::get_labels(self : Dataset) -> Array[Double] {
  self.labels
}

///|
/// 获取指定范围的子集
pub fn Dataset::slice(self : Dataset, start : Int, end : Int) -> Dataset {
  if start < 0 || start >= self.len() {
    abort("起始索引越界")
  }
  if end < start || end > self.len() {
    abort("结束索引越界")
  }
  let n_samples = end - start
  let new_features = Array::make(n_samples, [])
  let new_labels = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    new_features[i] = self.features[start + i]
    new_labels[i] = self.labels[start + i]
  }
  Dataset::new(new_features, new_labels)
}

///|
/// 获取指定范围的子集（语义同 slice）
pub fn Dataset::range(self : Dataset, start : Int, end : Int) -> Dataset {
  self.slice(start, end)
}

///|
/// 映射数据集
pub fn Dataset::map(
  self : Dataset,
  mapper : (Array[Double], Double) -> (Array[Double], Double),
) -> Dataset {
  let n_samples = self.len()
  let new_features = Array::make(n_samples, [])
  let new_labels = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let (features, label) = mapper(self.features[i], self.labels[i])
    new_features[i] = features
    new_labels[i] = label
  }
  Dataset::new(new_features, new_labels)
}

///|
/// 打乱数据集
pub fn Dataset::shuffle(self : Dataset, seed : Int) -> Dataset {
  let n_samples = self.len()
  if n_samples <= 1 {
    return Dataset::new(self.features, self.labels)
  }
  let indices = Array::make(n_samples, 0)
  for i = 0; i < n_samples; i = i + 1 {
    indices[i] = i
  }
  shuffle_array(indices, seed)
  let new_features = Array::make(n_samples, [])
  let new_labels = Array::make(n_samples, 0.0)
  for i = 0; i < n_samples; i = i + 1 {
    let idx = indices[i]
    new_features[i] = self.features[idx]
    new_labels[i] = self.labels[idx]
  }
  Dataset::new(new_features, new_labels)
}

///|
/// 数据分割函数
/// 将数据集分割为训练集和测试集
pub fn train_test_split(
  features : Array[Array[Double]],
  labels : Array[Double],
  test_size? : Double = 0.2,
  shuffle? : Bool = true,
  random_seed? : Int = 42,
) -> (Dataset, Dataset) {
  let n_samples = features.length()
  if n_samples == 0 {
    abort("特征不能为空")
  }
  validate_features_and_labels(features, labels)
  if test_size <= 0.0 || test_size >= 1.0 {
    abort("test_size 必须在 (0, 1) 之间")
  }

  // 创建索引数组
  let indices = Array::make(n_samples, 0)
  for i = 0; i < n_samples; i = i + 1 {
    indices[i] = i
  }

  // 如果需要打乱
  if shuffle {
    shuffle_array(indices, random_seed)
  }

  // 计算分割点
  let n_test = (n_samples.to_double() * test_size).to_int()
  let n_train = n_samples - n_test

  // 分割数据
  let train_features = Array::make(n_train, [])
  let train_labels = Array::make(n_train, 0.0)
  let test_features = Array::make(n_test, [])
  let test_labels = Array::make(n_test, 0.0)
  for i = 0; i < n_train; i = i + 1 {
    train_features[i] = features[indices[i]]
    train_labels[i] = labels[indices[i]]
  }
  for i = 0; i < n_test; i = i + 1 {
    test_features[i] = features[indices[n_train + i]]
    test_labels[i] = labels[indices[n_train + i]]
  }
  (
    Dataset::new(train_features, train_labels),
    Dataset::new(test_features, test_labels),
  )
}

///|
/// K折交叉验证数据分割
pub fn k_fold_split(
  features : Array[Array[Double]],
  labels : Array[Double],
  n_splits : Int,
) -> Array[(Dataset, Dataset)] {
  let n_samples = features.length()
  if n_samples == 0 {
    abort("特征不能为空")
  }
  validate_features_and_labels(features, labels)
  if n_splits <= 1 || n_splits > n_samples {
    abort("n_splits 必须在 (1, n_samples] 之间")
  }
  let fold_size = n_samples / n_splits
  let folds = Array::make(
    n_splits,
    (Dataset::new([], []), Dataset::new([], [])),
  )
  for i = 0; i < n_splits; i = i + 1 {
    let test_start = i * fold_size
    let test_end = if i == n_splits - 1 {
      n_samples
    } else {
      (i + 1) * fold_size
    }
    let n_test = test_end - test_start
    let n_train = n_samples - n_test
    let train_features = Array::make(n_train, [])
    let train_labels = Array::make(n_train, 0.0)
    let test_features = Array::make(n_test, [])
    let test_labels = Array::make(n_test, 0.0)
    let mut train_idx = 0
    for j = 0; j < n_samples; j = j + 1 {
      if j >= test_start && j < test_end {
        test_features[j - test_start] = features[j]
        test_labels[j - test_start] = labels[j]
      } else {
        train_features[train_idx] = features[j]
        train_labels[train_idx] = labels[j]
        train_idx = train_idx + 1
      }
    }
    folds[i] = (
      Dataset::new(train_features, train_labels),
      Dataset::new(test_features, test_labels),
    )
  }
  folds
}

///|
/// 辅助函数：打乱数组（Fisher-Yates 洗牌算法）
fn shuffle_array(arr : Array[Int], seed : Int) -> Unit {
  let n = arr.length()
  let mut rng_state = seed
  for i = n - 1; i > 0; i = i - 1 {
    // 简单的线性同余生成器
    rng_state = (rng_state * 1103515245 + 12345) % 2147483647
    let j = (rng_state % (i + 1)).abs()

    // 交换 arr[i] 和 arr[j]
    let temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp
  }
}
