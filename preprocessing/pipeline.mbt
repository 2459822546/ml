///|
/// 简单流水线与列变换器，串联预处理步骤

///|
/// 内部校验：检查矩阵非空且列数一致
fn validate_matrix(data : Array[Array[Double]]) -> (Int, Int) {
  if data.length() == 0 {
    abort("数据不能为空")
  }
  let n_features = data[0].length()
  for i = 0; i < data.length(); i = i + 1 {
    if data[i].length() != n_features {
      abort("第 \{i} 行的特征数量与其他行不一致")
    }
  }
  (data.length(), n_features)
}

///|
/// 从矩阵提取指定列
fn select_columns(
  data : Array[Array[Double]],
  cols : Array[Int],
) -> Array[Array[Double]] {
  let (n_samples, _) = validate_matrix(data)
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(cols.length(), 0.0)
    for j = 0; j < cols.length(); j = j + 1 {
      row[j] = data[i][cols[j]]
    }
    result[i] = row
  }
  result
}

///|
/// 将多个特征块按列拼接
fn concat_blocks(blocks : Array[Array[Array[Double]]]) -> Array[Array[Double]] {
  if blocks.length() == 0 {
    return []
  }
  let n_samples = blocks[0].length()
  if n_samples == 0 {
    abort("数据不能为空")
  }
  let mut total_features = 0
  for b = 0; b < blocks.length(); b = b + 1 {
    if blocks[b].length() != n_samples {
      abort("特征块的样本数不一致")
    }
    total_features = total_features + blocks[b][0].length()
  }
  let out = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(total_features, 0.0)
    let mut offset = 0
    for b = 0; b < blocks.length(); b = b + 1 {
      let block = blocks[b]
      for j = 0; j < block[i].length(); j = j + 1 {
        row[offset + j] = block[i][j]
      }
      offset = offset + block[i].length()
    }
    out[i] = row
  }
  out
}

///|
/// 支持的变换器封装
pub enum Transformer {
  Standard(StandardScaler)
  MinMax(MinMaxScaler)
  Polynomial(PolynomialFeatures)
}

///|
pub fn transformer_standard(s : StandardScaler) -> Transformer {
  Standard(s)
}

///|
pub fn transformer_minmax(s : MinMaxScaler) -> Transformer {
  MinMax(s)
}

///|
pub fn transformer_polynomial(p : PolynomialFeatures) -> Transformer {
  Polynomial(p)
}

///|
fn transformer_fit(t : Transformer, data : Array[Array[Double]]) -> Transformer {
  match t {
    Standard(s) => {
      let sc = s
      sc.fit(data)
      Standard(sc)
    }
    MinMax(s) => {
      let mm = s
      mm.fit(data)
      MinMax(mm)
    }
    Polynomial(p) => {
      let poly = p
      poly.fit(data)
      Polynomial(poly)
    }
  }
}

///|
fn transformer_transform(
  t : Transformer,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  match t {
    Standard(s) => s.transform(data)
    MinMax(s) => s.transform(data)
    Polynomial(p) => p.transform(data)
  }
}

///|
fn transformer_fit_transform(
  t : Transformer,
  data : Array[Array[Double]],
) -> (Transformer, Array[Array[Double]]) {
  match t {
    Standard(s) => {
      let sc = s
      let out = sc.fit_transform(data)
      (Standard(sc), out)
    }
    MinMax(s) => {
      let mm = s
      let out = mm.fit_transform(data)
      (MinMax(mm), out)
    }
    Polynomial(p) => {
      let poly = p
      let out = poly.fit_transform(data)
      (Polynomial(poly), out)
    }
  }
}

///|
/// 顺序流水线：按步骤依次拟合并转换
pub struct Pipeline {
  steps : Array[(String, Transformer)]
}

///|
pub fn Pipeline::new() -> Pipeline {
  { steps: [] }
}

///|
/// 追加步骤
pub fn Pipeline::add_step(
  self : Pipeline,
  name : String,
  transformer : Transformer,
) -> Unit {
  self.steps.push((name, transformer))
}

///|
/// 拟合（不返回数据）
pub fn Pipeline::fit(self : Pipeline, data : Array[Array[Double]]) -> Unit {
  let _ = validate_matrix(data)
  let mut current = data
  for i = 0; i < self.steps.length(); i = i + 1 {
    let step = self.steps[i]
    let (name, transformer) = step
    let (fitted, output) = transformer_fit_transform(transformer, current)
    self.steps[i] = (name, fitted)
    current = output
  }
}

///|
/// 拟合并转换
pub fn Pipeline::fit_transform(
  self : Pipeline,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  let _ = validate_matrix(data)
  let mut current = data
  for i = 0; i < self.steps.length(); i = i + 1 {
    let step = self.steps[i]
    let (name, transformer) = step
    let (fitted, output) = transformer_fit_transform(transformer, current)
    self.steps[i] = (name, fitted)
    current = output
  }
  current
}

///|
/// 仅转换（要求已 fit）
pub fn Pipeline::transform(
  self : Pipeline,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  let _ = validate_matrix(data)
  let mut current = data
  for i = 0; i < self.steps.length(); i = i + 1 {
    current = transformer_transform(self.steps[i].1, current)
  }
  current
}

///|
/// 列变换器：对指定列应用不同变换器
pub struct ColumnTransformer {
  mut transformers : Array[(String, Array[Int], Transformer)]
  remainder_passthrough : Bool
  mut fitted : Bool
}

///|
pub fn ColumnTransformer::new(
  remainder_passthrough? : Bool = false,
) -> ColumnTransformer {
  { transformers: [], remainder_passthrough, fitted: false }
}

///|
/// 添加一组列的变换器
pub fn ColumnTransformer::add(
  self : ColumnTransformer,
  name : String,
  columns : Array[Int],
  transformer : Transformer,
) -> Unit {
  self.transformers.push((name, columns, transformer))
}

///|
fn compute_unused_columns(
  n_features : Int,
  used : Array[Array[Int]],
) -> Array[Int] {
  let marks = Array::make(n_features, false)
  for i = 0; i < used.length(); i = i + 1 {
    let cols = used[i]
    for j = 0; j < cols.length(); j = j + 1 {
      let idx = cols[j]
      if idx < 0 || idx >= n_features {
        abort("列索引超出范围")
      }
      marks[idx] = true
    }
  }
  let rest = []
  for j = 0; j < n_features; j = j + 1 {
    if not(marks[j]) {
      rest.push(j)
    }
  }
  rest
}

///|
fn transform_blocks(
  transformers : Array[(String, Array[Int], Transformer)],
  data : Array[Array[Double]],
) -> Array[(String, Array[Array[Double]])] {
  let blocks = Array::make(transformers.length(), ("", []))
  for i = 0; i < transformers.length(); i = i + 1 {
    let (name, cols, transformer) = transformers[i]
    let sliced = select_columns(data, cols)
    let out = transformer_transform(transformer, sliced)
    blocks[i] = (name, out)
  }
  blocks
}

///|
pub fn ColumnTransformer::fit(
  self : ColumnTransformer,
  data : Array[Array[Double]],
) -> Unit {
  let (_, n_features) = validate_matrix(data)
  let updated = Array::make(
    self.transformers.length(),
    ("", [], Standard(StandardScaler::new())),
  )
  for i = 0; i < self.transformers.length(); i = i + 1 {
    let (name, cols, transformer) = self.transformers[i]
    let _ = compute_unused_columns(n_features, [cols]) // 验证索引范围
    let sliced = select_columns(data, cols)
    let fitted = transformer_fit(transformer, sliced)
    updated[i] = (name, cols, fitted)
  }
  self.transformers = updated
  self.fitted = true
}

///|
pub fn ColumnTransformer::transform(
  self : ColumnTransformer,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let (_, n_features) = validate_matrix(data)
  let blocks = []
  // 已配置的变换块
  let transformed = transform_blocks(self.transformers, data)
  for i = 0; i < transformed.length(); i = i + 1 {
    blocks.push(transformed[i].1)
  }

  // remainder
  if self.remainder_passthrough {
    let used_cols = Array::make(self.transformers.length(), [])
    for i = 0; i < self.transformers.length(); i = i + 1 {
      used_cols[i] = self.transformers[i].1
    }
    let remaining = compute_unused_columns(n_features, used_cols)
    if remaining.length() > 0 {
      let rest_block = select_columns(data, remaining)
      blocks.push(rest_block)
    }
  }
  concat_blocks(blocks)
}

///|
pub fn ColumnTransformer::fit_transform(
  self : ColumnTransformer,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  self.fit(data)
  self.transform(data)
}
