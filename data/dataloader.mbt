///|
/// DataLoader - 数据加载器
/// 用于批量加载数据，支持打乱和批处理
pub struct DataLoader {
  dataset : Dataset
  batch_size : Int
  shuffle : Bool
  drop_last : Bool
  base_seed : Int
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
) -> DataLoader {
  if batch_size <= 0 {
    abort("batch_size 必须大于 0")
  }
  if dataset.len() == 0 {
    abort("dataset 不能为空")
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
    shuffle_indices(self.indices, self.base_seed + self.current_epoch)
  }
  self.current_epoch = self.current_epoch + 1
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
  let batch_features = Array::make(actual_batch_size, [])
  let batch_labels = Array::make(actual_batch_size, 0.0)
  for i = 0; i < actual_batch_size; i = i + 1 {
    let idx = self.indices[start_idx + i]
    let (features, label) = self.dataset.get(idx)
    batch_features[i] = features
    batch_labels[i] = label
  }
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
