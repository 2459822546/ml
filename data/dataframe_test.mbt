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

///|
test "DataFrame - filter" {
  let columns = ["A", "B"]
  let data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
  let df = DataFrame::new(columns, data)
  let filtered = df.filter(fn(row) { row[0] > 2.0 })
  assert_eq(filtered.n_rows(), 2)
  assert_eq(filtered.get_row(0)[0], 3.0)
  assert_eq(filtered.get_row(1)[0], 5.0)
}

///|
test "DataFrame - sort" {
  let columns = ["A", "B"]
  let data = [[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]]
  let df = DataFrame::new(columns, data)
  let sorted = df.sort("B", ascending=true)
  assert_eq(sorted.get_row(0)[1], 1.0)
  assert_eq(sorted.get_row(2)[1], 3.0)
}

///|
test "DataFrame - drop and rename" {
  let columns = ["A", "B", "C"]
  let data = [[1.0, 2.0, 3.0]]
  let df = DataFrame::new(columns, data)
  let dropped = df.drop(["B"])
  assert_eq(dropped.n_cols(), 2)
  assert_eq(dropped.get_columns()[0], "A")
  assert_eq(dropped.get_columns()[1], "C")
  let renamed = dropped.rename(["A"], ["X"])
  assert_eq(renamed.get_columns()[0], "X")
}

///|
test "DataFrame - fillna" {
  let columns = ["A", "B"]
  let nan = @double.not_a_number
  let data = [[1.0, nan], [nan, 4.0]]
  let df = DataFrame::new(columns, data)
  let filled = df.fillna(0.0)
  assert_eq(filled.get_row(0)[1], 0.0)
  assert_eq(filled.get_row(1)[0], 0.0)
}

///|
test "DataFrame - describe" {
  let columns = ["A", "B"]
  let data = [[1.0, 2.0], [3.0, 4.0]]
  let df = DataFrame::new(columns, data)
  let stats = df.describe()
  assert_eq(stats.n_rows, 2)
  assert_eq(stats.n_cols, 2)
  assert_eq(stats.columns[0], "A")
  assert_eq(stats.stats[0].count, 2)
  assert_true((stats.stats[0].mean - 2.0).abs() < 1.0e-9)
  assert_true((stats.stats[0].std - 1.0).abs() < 1.0e-9)
  assert_true((stats.stats[0].variance - 1.0).abs() < 1.0e-9)
  assert_eq(stats.stats[0].min, 1.0)
  assert_true((stats.stats[0].q25 - 1.5).abs() < 1.0e-9)
  assert_true((stats.stats[0].median - 2.0).abs() < 1.0e-9)
  assert_true((stats.stats[0].q75 - 2.5).abs() < 1.0e-9)
  assert_eq(stats.stats[0].max, 3.0)
  let table = stats.to_table()
  assert_eq(table.length(), 10)
  assert_eq(table[0][0], "stat")
  assert_eq(table[1][0], "count")
}
