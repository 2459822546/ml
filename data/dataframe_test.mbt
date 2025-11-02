///|
/// DataFrame 测试
test "DataFrame - 基本创建和访问" {
  let columns = ["A", "B", "C"]
  let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  let df = DataFrame::new(columns, data)
  assert_eq(df.n_rows(), 3)
  assert_eq(df.n_cols(), 3)
  assert_eq(df.shape(), (3, 3))
}

///|
test "DataFrame - 获取行和列" {
  let columns = ["X", "Y"]
  let data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
  let df = DataFrame::new(columns, data)

  // 获取行
  let row = df.get_row(1)
  assert_true((row[0] - 3.0).abs() < 0.0001)
  assert_true((row[1] - 4.0).abs() < 0.0001)

  // 获取列
  let col = df.get_column("X")
  assert_eq(col.length(), 3)
  assert_true((col[0] - 1.0).abs() < 0.0001)
  assert_true((col[1] - 3.0).abs() < 0.0001)
  assert_true((col[2] - 5.0).abs() < 0.0001)
}

///|
test "DataFrame - 切片操作" {
  let columns = ["A", "B"]
  let data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
  let df = DataFrame::new(columns, data)

  // head
  let df_head = df.head(2)
  assert_eq(df_head.n_rows(), 2)

  // tail
  let df_tail = df.tail(2)
  assert_eq(df_tail.n_rows(), 2)

  // slice
  let df_slice = df.slice(1, 3)
  assert_eq(df_slice.n_rows(), 2)
}

///|
test "DataFrame - 选择列" {
  let columns = ["A", "B", "C"]
  let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  let df = DataFrame::new(columns, data)
  let df_selected = df.select(["A", "C"])
  assert_eq(df_selected.n_cols(), 2)
  assert_eq(df_selected.get_columns()[0], "A")
  assert_eq(df_selected.get_columns()[1], "C")
}

///|
test "DataFrame - 添加行" {
  let columns = ["X", "Y"]
  let data = [[1.0, 2.0]]
  let df = DataFrame::new(columns, data)
  assert_eq(df.n_rows(), 1)
  df.add_row([3.0, 4.0])
  assert_eq(df.n_rows(), 2)
  let row = df.get_row(1)
  assert_true((row[0] - 3.0).abs() < 0.0001)
  assert_true((row[1] - 4.0).abs() < 0.0001)
}

///|
test "DataFrame - 空 DataFrame" {
  let columns = ["A", "B"]
  let df = DataFrame::empty(columns)
  assert_eq(df.n_rows(), 0)
  assert_eq(df.n_cols(), 2)
}
