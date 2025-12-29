///|
/// DataFrame - 表格数据结构
/// 类似 pandas DataFrame，用于处理结构化数据
pub struct DataFrame {
  columns : Array[String] // 列名
  mut data : Array[Array[Double]] // 数据，每行是一个样本
  mut index : Array[Int] // 行索引
}

///|
/// 列统计结果
pub struct ColumnStats {
  count : Int
  mean : Double
  std : Double
  min : Double
  max : Double
}

///|
/// DataFrame 统计结果
pub struct DataFrameStats {
  n_rows : Int
  n_cols : Int
  columns : Array[String]
  stats : Array[ColumnStats]
}

///|
/// 校验数据行的列数是否与列名一致
fn validate_data_shape(
  columns : Array[String],
  data : Array[Array[Double]],
) -> Unit {
  let expected_cols = columns.length()
  for i = 0; i < data.length(); i = i + 1 {
    if data[i].length() != expected_cols {
      abort("第 \{i} 行的列数与列名不匹配")
    }
  }
}

///|
pub fn DataFrame::new(
  columns : Array[String],
  data : Array[Array[Double]],
) -> DataFrame {
  validate_data_shape(columns, data)
  let n_rows = data.length()
  let index = Array::make(n_rows, 0)
  for i = 0; i < n_rows; i = i + 1 {
    index[i] = i
  }
  { columns, data, index }
}

///|
/// 创建空的 DataFrame
pub fn DataFrame::empty(columns : Array[String]) -> DataFrame {
  { columns, data: [], index: [] }
}

///|
/// 获取行数
pub fn DataFrame::n_rows(self : DataFrame) -> Int {
  self.data.length()
}

///|
/// 获取列数
pub fn DataFrame::n_cols(self : DataFrame) -> Int {
  self.columns.length()
}

///|
/// 获取形状 (行数, 列数)
pub fn DataFrame::shape(self : DataFrame) -> (Int, Int) {
  (self.n_rows(), self.n_cols())
}

///|
/// 获取指定行
pub fn DataFrame::get_row(self : DataFrame, index : Int) -> Array[Double] {
  if index < 0 || index >= self.n_rows() {
    abort("行索引越界")
  }
  self.data[index]
}

///|
/// 获取指定列
pub fn DataFrame::get_column(
  self : DataFrame,
  col_name : String,
) -> Array[Double] {
  let col_index = self.find_column_index(col_name)
  if col_index == -1 {
    abort("列名不存在: " + col_name)
  }
  let result = Array::make(self.n_rows(), 0.0)
  for i = 0; i < self.n_rows(); i = i + 1 {
    result[i] = self.data[i][col_index]
  }
  result
}

///|
/// 通过索引获取列
pub fn DataFrame::get_column_by_index(
  self : DataFrame,
  col_index : Int,
) -> Array[Double] {
  if col_index < 0 || col_index >= self.n_cols() {
    abort("列索引越界")
  }
  let result = Array::make(self.n_rows(), 0.0)
  for i = 0; i < self.n_rows(); i = i + 1 {
    result[i] = self.data[i][col_index]
  }
  result
}

///|
/// 设置指定单元格的值
pub fn DataFrame::set(
  self : DataFrame,
  row : Int,
  col : Int,
  value : Double,
) -> Unit {
  if row < 0 || row >= self.n_rows() {
    abort("行索引越界")
  }
  if col < 0 || col >= self.n_cols() {
    abort("列索引越界")
  }
  self.data[row][col] = value
}

///|
/// 添加一行数据
pub fn DataFrame::add_row(self : DataFrame, row : Array[Double]) -> Unit {
  if row.length() != self.n_cols() {
    abort("行数据列数不匹配")
  }

  // 扩展数据数组
  let new_data = Array::make(self.n_rows() + 1, [])
  for i = 0; i < self.n_rows(); i = i + 1 {
    new_data[i] = self.data[i]
  }
  new_data[self.n_rows()] = row
  self.data = new_data

  // 扩展索引
  let new_index = Array::make(self.n_rows() + 1, 0)
  for i = 0; i < self.index.length(); i = i + 1 {
    new_index[i] = self.index[i]
  }
  new_index[self.index.length()] = self.index.length()
  self.index = new_index
}

///|
/// 选择多列，返回新的 DataFrame
pub fn DataFrame::select(
  self : DataFrame,
  col_names : Array[String],
) -> DataFrame {
  let col_indices = []
  for i = 0; i < col_names.length(); i = i + 1 {
    let idx = self.find_column_index(col_names[i])
    if idx == -1 {
      abort("列名不存在: " + col_names[i])
    }
    col_indices.push(idx)
  }

  // 创建新数据
  let new_data = Array::make(self.n_rows(), [])
  for i = 0; i < self.n_rows(); i = i + 1 {
    let row = Array::make(col_names.length(), 0.0)
    for j = 0; j < col_names.length(); j = j + 1 {
      row[j] = self.data[i][col_indices[j]]
    }
    new_data[i] = row
  }
  DataFrame::new(col_names, new_data)
}

///|
/// 选择行范围，返回新的 DataFrame
pub fn DataFrame::slice(self : DataFrame, start : Int, end : Int) -> DataFrame {
  if start < 0 || start >= self.n_rows() {
    abort("起始索引越界")
  }
  if end < start || end > self.n_rows() {
    abort("结束索引越界")
  }
  let n_rows = end - start
  let new_data = Array::make(n_rows, [])
  for i = 0; i < n_rows; i = i + 1 {
    new_data[i] = self.data[start + i]
  }
  DataFrame::new(self.columns, new_data)
}

///|
/// 获取前 n 行
pub fn DataFrame::head(self : DataFrame, n : Int) -> DataFrame {
  let actual_n = if n > self.n_rows() { self.n_rows() } else { n }
  self.slice(0, actual_n)
}

///|
/// 获取后 n 行
pub fn DataFrame::tail(self : DataFrame, n : Int) -> DataFrame {
  let actual_n = if n > self.n_rows() { self.n_rows() } else { n }
  self.slice(self.n_rows() - actual_n, self.n_rows())
}

///|
/// 辅助函数：查找列名对应的索引
fn DataFrame::find_column_index(self : DataFrame, col_name : String) -> Int {
  for i = 0; i < self.columns.length(); i = i + 1 {
    if self.columns[i] == col_name {
      return i
    }
  }
  -1
}

///|
/// 转换为 Array[Array[Double]]
pub fn DataFrame::to_array(self : DataFrame) -> Array[Array[Double]] {
  self.data
}

///|
/// 获取列名
pub fn DataFrame::get_columns(self : DataFrame) -> Array[String] {
  self.columns
}

///|
/// 描述性统计（返回结果）
pub fn DataFrame::describe(self : DataFrame) -> DataFrameStats {
  let n_rows = self.n_rows()
  let n_cols = self.n_cols()
  let stats = Array::make(n_cols, {
    count: 0,
    mean: @double.not_a_number,
    std: @double.not_a_number,
    min: @double.not_a_number,
    max: @double.not_a_number,
  })
  if n_rows == 0 || n_cols == 0 {
    return { n_rows, n_cols, columns: self.columns, stats }
  }
  for col_idx = 0; col_idx < n_cols; col_idx = col_idx + 1 {
    let mut sum = 0.0
    let mut count = 0
    let mut min_val = @double.not_a_number
    let mut max_val = @double.not_a_number
    for row_idx = 0; row_idx < n_rows; row_idx = row_idx + 1 {
      let value = self.data[row_idx][col_idx]
      if is_nan(value) {
        continue
      }
      sum = sum + value
      if count == 0 {
        min_val = value
        max_val = value
      } else {
        if value < min_val {
          min_val = value
        }
        if value > max_val {
          max_val = value
        }
      }
      count = count + 1
    }
    if count == 0 {
      stats[col_idx] = {
        count: 0,
        mean: @double.not_a_number,
        std: @double.not_a_number,
        min: @double.not_a_number,
        max: @double.not_a_number,
      }
      continue
    }
    let mean = sum / count.to_double()
    let mut var_sum = 0.0
    for row_idx = 0; row_idx < n_rows; row_idx = row_idx + 1 {
      let value = self.data[row_idx][col_idx]
      if is_nan(value) {
        continue
      }
      let diff = value - mean
      var_sum = var_sum + diff * diff
    }
    let std = (var_sum / count.to_double()).sqrt()
    stats[col_idx] = { count, mean, std, min: min_val, max: max_val }
  }
  { n_rows, n_cols, columns: self.columns, stats }
}

///|
/// 行过滤
pub fn DataFrame::filter(
  self : DataFrame,
  predicate : (Array[Double]) -> Bool,
) -> DataFrame {
  let rows = []
  for i = 0; i < self.n_rows(); i = i + 1 {
    let row = self.data[i]
    if predicate(row) {
      rows.push(row)
    }
  }
  let new_data = Array::make(rows.length(), [])
  for i = 0; i < rows.length(); i = i + 1 {
    new_data[i] = rows[i]
  }
  DataFrame::new(self.columns, new_data)
}

///|
/// 按列排序
pub fn DataFrame::sort(
  self : DataFrame,
  col_name : String,
  ascending? : Bool = true,
) -> DataFrame {
  let col_index = self.find_column_index(col_name)
  if col_index == -1 {
    abort("列名不存在: " + col_name)
  }
  let n = self.n_rows()
  let indices = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    indices[i] = i
  }
  for i = 1; i < n; i = i + 1 {
    let key = indices[i]
    let mut j = i - 1
    while j >= 0 {
      let a = self.data[indices[j]][col_index]
      let b = self.data[key][col_index]
      let cmp = compare_values(a, b, ascending)
      if cmp <= 0 {
        break
      }
      indices[j + 1] = indices[j]
      j = j - 1
    }
    indices[j + 1] = key
  }
  let new_data = Array::make(n, [])
  for i = 0; i < n; i = i + 1 {
    new_data[i] = self.data[indices[i]]
  }
  DataFrame::new(self.columns, new_data)
}

///|
/// 删除列
pub fn DataFrame::drop(
  self : DataFrame,
  col_names : Array[String],
) -> DataFrame {
  if col_names.length() == 0 {
    return DataFrame::new(self.columns, self.data)
  }
  let drop_mask = Array::make(self.n_cols(), false)
  for i = 0; i < col_names.length(); i = i + 1 {
    let idx = self.find_column_index(col_names[i])
    if idx == -1 {
      abort("列名不存在: " + col_names[i])
    }
    drop_mask[idx] = true
  }
  let keep_cols = []
  let keep_indices = []
  for i = 0; i < self.n_cols(); i = i + 1 {
    if not(drop_mask[i]) {
      keep_cols.push(self.columns[i])
      keep_indices.push(i)
    }
  }
  let new_columns = Array::make(keep_cols.length(), "")
  for i = 0; i < keep_cols.length(); i = i + 1 {
    new_columns[i] = keep_cols[i]
  }
  let new_data = Array::make(self.n_rows(), [])
  for i = 0; i < self.n_rows(); i = i + 1 {
    let row = Array::make(keep_indices.length(), 0.0)
    for j = 0; j < keep_indices.length(); j = j + 1 {
      row[j] = self.data[i][keep_indices[j]]
    }
    new_data[i] = row
  }
  DataFrame::new(new_columns, new_data)
}

///|
/// 重命名列
pub fn DataFrame::rename(
  self : DataFrame,
  old_names : Array[String],
  new_names : Array[String],
) -> DataFrame {
  if old_names.length() != new_names.length() {
    abort("列名数量不匹配")
  }
  let new_columns = Array::make(self.n_cols(), "")
  for i = 0; i < self.n_cols(); i = i + 1 {
    new_columns[i] = self.columns[i]
  }
  for i = 0; i < old_names.length(); i = i + 1 {
    let idx = self.find_column_index(old_names[i])
    if idx == -1 {
      abort("列名不存在: " + old_names[i])
    }
    new_columns[idx] = new_names[i]
  }
  DataFrame::new(new_columns, self.data)
}

///|
/// 填充缺失值（NaN）
pub fn DataFrame::fillna(self : DataFrame, value : Double) -> DataFrame {
  let new_data = Array::make(self.n_rows(), [])
  for i = 0; i < self.n_rows(); i = i + 1 {
    let row = Array::make(self.n_cols(), 0.0)
    for j = 0; j < self.n_cols(); j = j + 1 {
      let v = self.data[i][j]
      row[j] = if is_nan(v) { value } else { v }
    }
    new_data[i] = row
  }
  DataFrame::new(self.columns, new_data)
}

///|
/// NaN 判断
fn is_nan(value : Double) -> Bool {
  value != value
}

///|
/// 数值比较（支持 NaN）
fn compare_values(a : Double, b : Double, ascending : Bool) -> Int {
  let a_nan = is_nan(a)
  let b_nan = is_nan(b)
  if a_nan && b_nan {
    return 0
  }
  if a_nan {
    return if ascending { 1 } else { -1 }
  }
  if b_nan {
    return if ascending { -1 } else { 1 }
  }
  if a == b {
    return 0
  }
  if ascending {
    if a < b {
      -1
    } else {
      1
    }
  } else if a > b {
    -1
  } else {
    1
  }
}
