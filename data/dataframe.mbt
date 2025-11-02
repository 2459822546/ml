///|
/// DataFrame - 表格数据结构
/// 类似 pandas DataFrame，用于处理结构化数据
pub struct DataFrame {
  columns : Array[String] // 列名
  mut data : Array[Array[Double]] // 数据，每行是一个样本
  mut index : Array[Int] // 行索引
}

///|
pub fn DataFrame::new(
  columns : Array[String],
  data : Array[Array[Double]],
) -> DataFrame {
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
pub fn n_rows(self : DataFrame) -> Int {
  self.data.length()
}

///|
/// 获取列数
pub fn n_cols(self : DataFrame) -> Int {
  self.columns.length()
}

///|
/// 获取形状 (行数, 列数)
pub fn shape(self : DataFrame) -> (Int, Int) {
  (self.n_rows(), self.n_cols())
}

///|
/// 获取指定行
pub fn get_row(self : DataFrame, index : Int) -> Array[Double] {
  if index < 0 || index >= self.n_rows() {
    abort("行索引越界")
  }
  self.data[index]
}

///|
/// 获取指定列
pub fn get_column(self : DataFrame, col_name : String) -> Array[Double] {
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
pub fn get_column_by_index(self : DataFrame, col_index : Int) -> Array[Double] {
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
pub fn set(self : DataFrame, row : Int, col : Int, value : Double) -> Unit {
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
pub fn add_row(self : DataFrame, row : Array[Double]) -> Unit {
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
pub fn select(self : DataFrame, col_names : Array[String]) -> DataFrame {
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
pub fn slice(self : DataFrame, start : Int, end : Int) -> DataFrame {
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
pub fn head(self : DataFrame, n : Int) -> DataFrame {
  let actual_n = if n > self.n_rows() { self.n_rows() } else { n }
  self.slice(0, actual_n)
}

///|
/// 获取后 n 行
pub fn tail(self : DataFrame, n : Int) -> DataFrame {
  let actual_n = if n > self.n_rows() { self.n_rows() } else { n }
  self.slice(self.n_rows() - actual_n, self.n_rows())
}

///|
/// 辅助函数：查找列名对应的索引
fn find_column_index(self : DataFrame, col_name : String) -> Int {
  for i = 0; i < self.columns.length(); i = i + 1 {
    if self.columns[i] == col_name {
      return i
    }
  }
  -1
}

///|
/// 转换为 Array[Array[Double]]
pub fn to_array(self : DataFrame) -> Array[Array[Double]] {
  self.data
}

///|
/// 获取列名
pub fn get_columns(self : DataFrame) -> Array[String] {
  self.columns
}

///|
/// 描述性统计
pub fn describe(self : DataFrame) -> Unit {
  println("DataFrame 统计信息:")
  println("形状: (\{self.n_rows()}, \{self.n_cols()})")
  println("列名: \{self.columns}")
}
