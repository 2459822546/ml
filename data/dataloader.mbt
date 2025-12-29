///|
/// DataLoader - 数据加载器
/// 用于批量加载数据，支持打乱和批处理
/// Batch 规范：
/// - features 形状为 [batch, features]
/// - labels 为 Single([batch]) 或 Multi([batch, n_labels])
pub enum BatchLabels {
  Single(Array[Double])
  Multi(Array[Array[Double]])
}

///|
pub struct Batch {
  features : Array[Array[Double]]
  labels : BatchLabels
}

///|
fn validate_batch_features(
  features : Array[Array[Double]],
  batch_size : Int,
) -> Unit {
  if features.length() != batch_size {
    abort("batch features 和 labels 的样本数不匹配")
  }
  if features.length() == 0 {
    return
  }
  let expected_dim = features[0].length()
  for i = 0; i < features.length(); i = i + 1 {
    if features[i].length() != expected_dim {
      abort("batch 样本 \{i} 的特征维度不一致")
    }
  }
}

///|
pub fn Batch::new_single(
  features : Array[Array[Double]],
  labels : Array[Double],
) -> Batch {
  validate_batch_features(features, labels.length())
  { features, labels: Single(labels) }
}

///|
pub fn Batch::new_multi(
  features : Array[Array[Double]],
  labels : Array[Array[Double]],
) -> Batch {
  validate_batch_features(features, labels.length())
  if labels.length() == 0 {
    return { features, labels: Multi(labels) }
  }
  let expected_dim = labels[0].length()
  for i = 0; i < labels.length(); i = i + 1 {
    if labels[i].length() != expected_dim {
      abort("batch 样本 \{i} 的标签维度不一致")
    }
  }
  { features, labels: Multi(labels) }
}

///|
pub struct DataLoader {
  dataset : Dataset
  batch_size : Int
  shuffle : Bool
  drop_last : Bool
  mut base_seed : Int
  num_workers : Int
  mut current_epoch : Int
  indices : Array[Int]
  mut current_idx : Int
}

///|
pub fn DataLoader::new(
  dataset : Dataset,
  batch_size? : Int = 32,
  shuffle? : Bool = false,
  drop_last? : Bool = false,
  seed? : Int = 42,
  num_workers? : Int = 1,
) -> DataLoader {
  if batch_size <= 0 {
    abort("batch_size 必须大于 0")
  }
  if dataset.len() == 0 {
    abort("dataset 不能为空")
  }
  if num_workers != 1 {
    abort("当前版本仅支持单线程 num_workers=1")
  }
  let n_samples = dataset.len()
  let indices = Array::make(n_samples, 0)
  for i = 0; i < n_samples; i = i + 1 {
    indices[i] = i
  }
  if shuffle {
    shuffle_indices(indices, seed)
  }
  {
    dataset,
    batch_size,
    shuffle,
    drop_last,
    base_seed: seed,
    num_workers,
    current_epoch: 0,
    indices,
    current_idx: 0,
  }
}

///|
/// 获取总批次数
pub fn DataLoader::n_batches(self : DataLoader) -> Int {
  let n_samples = self.dataset.len()
  if self.drop_last {
    n_samples / self.batch_size
  } else {
    (n_samples + self.batch_size - 1) / self.batch_size
  }
}

///|
/// 重置迭代器
pub fn DataLoader::reset(self : DataLoader) -> Unit {
  self.current_idx = 0
  if self.shuffle {
    shuffle_indices(self.indices, self.epoch_seed())
  }
  self.current_epoch = self.current_epoch + 1
}

///|
/// 设置随机种子
pub fn DataLoader::set_seed(self : DataLoader, seed : Int) -> Unit {
  self.base_seed = seed
  self.current_epoch = 0
  self.current_idx = 0
  if self.shuffle {
    shuffle_indices(self.indices, self.base_seed)
  }
}

///|
/// 获取当前 epoch 的种子
pub fn DataLoader::epoch_seed(self : DataLoader) -> Int {
  self.base_seed + self.current_epoch
}

///|
/// 获取下一批数据
pub fn DataLoader::next_batch(
  self : DataLoader,
) -> (Array[Array[Double]], Array[Double])? {
  if self.current_idx >= self.dataset.len() {
    return None
  }
  let start_idx = self.current_idx
  let end_idx = {
    let end = start_idx + self.batch_size
    if end > self.dataset.len() {
      self.dataset.len()
    } else {
      end
    }
  }
  if self.drop_last && end_idx - start_idx < self.batch_size {
    self.current_idx = self.dataset.len()
    return None
  }
  let actual_batch_size = end_idx - start_idx
  let batch_indices = Array::make(actual_batch_size, 0)
  for i = 0; i < actual_batch_size; i = i + 1 {
    batch_indices[i] = self.indices[start_idx + i]
  }
  let (batch_features, batch_labels) = self.dataset.get_batch(batch_indices)
  self.current_idx = end_idx
  Some((batch_features, batch_labels))
}

///|
/// 获取所有批次
pub fn DataLoader::all_batches(
  self : DataLoader,
) -> Array[(Array[Array[Double]], Array[Double])] {
  self.reset()
  let batches = []
  while true {
    match self.next_batch() {
      Some(batch) => batches.push(batch)
      None => break
    }
  }

  // 转换为固定数组
  let result = Array::make(batches.length(), ([], []))
  for i = 0; i < batches.length(); i = i + 1 {
    result[i] = batches[i]
  }
  result
}

///|
/// 遍历所有批次（函数式接口）
pub fn DataLoader::for_each_batch(
  self : DataLoader,
  f : (Array[Array[Double]], Array[Double]) -> Unit,
) -> Unit {
  self.reset()
  while true {
    match self.next_batch() {
      Some((features, labels)) => f(features, labels)
      None => break
    }
  }
}

///|
/// 辅助函数：打乱索引
fn shuffle_indices(indices : Array[Int], seed : Int) -> Unit {
  let n = indices.length()
  let mut rng_state = seed + 12345
  for i = n - 1; i > 0; i = i - 1 {
    rng_state = (rng_state * 1103515245 + 12345) % 2147483647
    let j = (rng_state % (i + 1)).abs()
    let temp = indices[i]
    indices[i] = indices[j]
    indices[j] = temp
  }
}

///|
/// BatchIterator - 简单的批处理迭代器
/// 不依赖 Dataset，直接处理数据数组
pub struct BatchIterator {
  data : Array[Array[Double]]
  batch_size : Int
  mut current_idx : Int
}

///|
pub fn BatchIterator::new(
  data : Array[Array[Double]],
  batch_size : Int,
) -> BatchIterator {
  if batch_size <= 0 {
    abort("batch_size 必须大于 0")
  }
  { data, batch_size, current_idx: 0 }
}

///|
/// 获取总批次数
pub fn BatchIterator::n_batches(self : BatchIterator) -> Int {
  (self.data.length() + self.batch_size - 1) / self.batch_size
}

///|
/// 重置迭代器
pub fn BatchIterator::reset(self : BatchIterator) -> Unit {
  self.current_idx = 0
}

///|
/// 获取下一批数据
pub fn BatchIterator::next(self : BatchIterator) -> Array[Array[Double]]? {
  if self.current_idx >= self.data.length() {
    return None
  }
  let start_idx = self.current_idx
  let end_idx = {
    let end = start_idx + self.batch_size
    if end > self.data.length() {
      self.data.length()
    } else {
      end
    }
  }
  let actual_batch_size = end_idx - start_idx
  let batch = Array::make(actual_batch_size, [])
  for i = 0; i < actual_batch_size; i = i + 1 {
    batch[i] = self.data[start_idx + i]
  }
  self.current_idx = end_idx
  Some(batch)
}

///|
/// 获取所有批次
pub fn BatchIterator::all_batches(
  self : BatchIterator,
) -> Array[Array[Array[Double]]] {
  self.reset()
  let batches = []
  while true {
    match self.next() {
      Some(batch) => batches.push(batch)
      None => break
    }
  }
  let result = Array::make(batches.length(), [])
  for i = 0; i < batches.length(); i = i + 1 {
    result[i] = batches[i]
  }
  result
}

///|
/// DataLoader 迭代器封装，便于 for 循环式使用
pub struct DataLoaderIter {
  loader : DataLoader
}

///|
pub fn DataLoaderIter::new(loader : DataLoader) -> DataLoaderIter {
  let l = loader
  l.reset()
  { loader: l }
}

///|
pub fn DataLoaderIter::next(
  self : DataLoaderIter,
) -> (Array[Array[Double]], Array[Double])? {
  self.loader.next_batch()
}

///|
pub fn DataLoader::iter(self : DataLoader) -> DataLoaderIter {
  DataLoaderIter::new(self)
}

///|
/// MultiDataLoader - 多标签数据加载器
pub struct MultiDataLoader {
  dataset : MultiDataset
  batch_size : Int
  shuffle : Bool
  drop_last : Bool
  mut base_seed : Int
  num_workers : Int
  mut current_epoch : Int
  indices : Array[Int]
  mut current_idx : Int
}

///|
pub fn MultiDataLoader::new(
  dataset : MultiDataset,
  batch_size? : Int = 32,
  shuffle? : Bool = false,
  drop_last? : Bool = false,
  seed? : Int = 42,
  num_workers? : Int = 1,
) -> MultiDataLoader {
  if batch_size <= 0 {
    abort("batch_size 必须大于 0")
  }
  if dataset.len() == 0 {
    abort("dataset 不能为空")
  }
  if num_workers != 1 {
    abort("当前版本仅支持单线程 num_workers=1")
  }
  let n_samples = dataset.len()
  let indices = Array::make(n_samples, 0)
  for i = 0; i < n_samples; i = i + 1 {
    indices[i] = i
  }
  if shuffle {
    shuffle_indices(indices, seed)
  }
  {
    dataset,
    batch_size,
    shuffle,
    drop_last,
    base_seed: seed,
    num_workers,
    current_epoch: 0,
    indices,
    current_idx: 0,
  }
}

///|
/// 获取总批次数
pub fn MultiDataLoader::n_batches(self : MultiDataLoader) -> Int {
  let n_samples = self.dataset.len()
  if self.drop_last {
    n_samples / self.batch_size
  } else {
    (n_samples + self.batch_size - 1) / self.batch_size
  }
}

///|
/// 重置迭代器
pub fn MultiDataLoader::reset(self : MultiDataLoader) -> Unit {
  self.current_idx = 0
  if self.shuffle {
    shuffle_indices(self.indices, self.epoch_seed())
  }
  self.current_epoch = self.current_epoch + 1
}

///|
/// 设置随机种子
pub fn MultiDataLoader::set_seed(self : MultiDataLoader, seed : Int) -> Unit {
  self.base_seed = seed
  self.current_epoch = 0
  self.current_idx = 0
  if self.shuffle {
    shuffle_indices(self.indices, self.base_seed)
  }
}

///|
/// 获取当前 epoch 的种子
pub fn MultiDataLoader::epoch_seed(self : MultiDataLoader) -> Int {
  self.base_seed + self.current_epoch
}

///|
/// 获取下一批数据
pub fn MultiDataLoader::next_batch(
  self : MultiDataLoader,
) -> (Array[Array[Double]], Array[Array[Double]])? {
  if self.current_idx >= self.dataset.len() {
    return None
  }
  let start_idx = self.current_idx
  let end_idx = {
    let end = start_idx + self.batch_size
    if end > self.dataset.len() {
      self.dataset.len()
    } else {
      end
    }
  }
  if self.drop_last && end_idx - start_idx < self.batch_size {
    self.current_idx = self.dataset.len()
    return None
  }
  let actual_batch_size = end_idx - start_idx
  let batch_indices = Array::make(actual_batch_size, 0)
  for i = 0; i < actual_batch_size; i = i + 1 {
    batch_indices[i] = self.indices[start_idx + i]
  }
  let (batch_features, batch_labels) = self.dataset.get_batch(batch_indices)
  self.current_idx = end_idx
  Some((batch_features, batch_labels))
}

///|
/// 获取所有批次
pub fn MultiDataLoader::all_batches(
  self : MultiDataLoader,
) -> Array[(Array[Array[Double]], Array[Array[Double]])] {
  self.reset()
  let batches = []
  while true {
    match self.next_batch() {
      Some(batch) => batches.push(batch)
      None => break
    }
  }
  let result = Array::make(batches.length(), ([], []))
  for i = 0; i < batches.length(); i = i + 1 {
    result[i] = batches[i]
  }
  result
}

///|
/// 遍历所有批次（函数式接口）
pub fn MultiDataLoader::for_each_batch(
  self : MultiDataLoader,
  f : (Array[Array[Double]], Array[Array[Double]]) -> Unit,
) -> Unit {
  self.reset()
  while true {
    match self.next_batch() {
      Some((features, labels)) => f(features, labels)
      None => break
    }
  }
}

///|
/// MultiDataLoader 迭代器封装
pub struct MultiDataLoaderIter {
  loader : MultiDataLoader
}

///|
pub fn MultiDataLoaderIter::new(
  loader : MultiDataLoader,
) -> MultiDataLoaderIter {
  let l = loader
  l.reset()
  { loader: l }
}

///|
pub fn MultiDataLoaderIter::next(
  self : MultiDataLoaderIter,
) -> (Array[Array[Double]], Array[Array[Double]])? {
  self.loader.next_batch()
}

///|
pub fn MultiDataLoader::iter(self : MultiDataLoader) -> MultiDataLoaderIter {
  MultiDataLoaderIter::new(self)
}
