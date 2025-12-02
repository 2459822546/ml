///|
/// 字符串比较工具，用于排序
fn compare_strings(a : String, b : String) -> Int {
  let len_a = a.length()
  let len_b = b.length()
  let min_len = if len_a < len_b { len_a } else { len_b }
  for i = 0; i < min_len; i = i + 1 {
    let char_a = a[i]
    let char_b = b[i]
    if char_a < char_b {
      return -1
    } else if char_a > char_b {
      return 1
    }
  }

  // 前缀相同，比较长度
  if len_a < len_b {
    -1
  } else if len_a > len_b {
    1
  } else {
    0
  }
}

///|
/// 标签编码器 (LabelEncoder)
/// 将标签编码为整数值
pub struct LabelEncoder {
  mut classes : Array[String]
  mut class_to_index : Map[String, Int]
  mut fitted : Bool
}

///|
pub fn LabelEncoder::new() -> LabelEncoder {
  { classes: [], class_to_index: Map::new(), fitted: false }
}

///|
/// 拟合标签数据
/// @param labels 标签数组
pub fn LabelEncoder::fit(self : LabelEncoder, labels : Array[String]) -> Unit {
  if labels.length() == 0 {
    abort("标签不能为空")
  }

  // 提取唯一的类别到固定数组
  let unique_list = []
  for i = 0; i < labels.length(); i = i + 1 {
    let mut found = false
    for j = 0; j < unique_list.length(); j = j + 1 {
      if unique_list[j] == labels[i] {
        found = true
        break
      }
    }
    if not(found) {
      unique_list.push(labels[i])
    }
  }

  // 转换为固定数组并排序
  self.classes = Array::make(unique_list.length(), "")
  for i = 0; i < unique_list.length(); i = i + 1 {
    self.classes[i] = unique_list[i]
  }
  self.classes.sort_by(fn(a, b) { compare_strings(a, b) })

  // 构建映射
  self.class_to_index = Map::new()
  for i = 0; i < self.classes.length(); i = i + 1 {
    self.class_to_index.set(self.classes[i], i)
  }
  self.fitted = true
}

///|
/// 转换标签为整数
/// @param labels 待转换的标签
/// @return 整数编码
pub fn LabelEncoder::transform(
  self : LabelEncoder,
  labels : Array[String],
) -> Array[Int] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let result = Array::make(labels.length(), 0)
  for i = 0; i < labels.length(); i = i + 1 {
    match self.class_to_index.get(labels[i]) {
      Some(index) => result[i] = index
      None => abort("未知的类别: " + labels[i])
    }
  }
  result
}

///|
/// 拟合并转换
pub fn LabelEncoder::fit_transform(
  self : LabelEncoder,
  labels : Array[String],
) -> Array[Int] {
  self.fit(labels)
  self.transform(labels)
}

///|
/// 反转换（整数转回标签）
pub fn LabelEncoder::inverse_transform(
  self : LabelEncoder,
  encoded : Array[Int],
) -> Array[String] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let result = Array::make(encoded.length(), "")
  for i = 0; i < encoded.length(); i = i + 1 {
    if encoded[i] < 0 || encoded[i] >= self.classes.length() {
      abort("无效的编码值: " + encoded[i].to_string())
    }
    result[i] = self.classes[encoded[i]]
  }
  result
}

///|
/// 获取所有类别
pub fn LabelEncoder::get_classes(self : LabelEncoder) -> Array[String] {
  self.classes
}

///|
/// 独热编码器 (OneHotEncoder)
/// 将分类特征转换为独热编码
pub struct OneHotEncoder {
  mut categories : Array[Array[String]] // 每个特征的类别
  mut fitted : Bool
}

///|
pub fn OneHotEncoder::new() -> OneHotEncoder {
  { categories: [], fitted: false }
}

///|
/// 拟合数据
/// @param data 数据，每行是一个样本，每列是一个特征
pub fn OneHotEncoder::fit(
  self : OneHotEncoder,
  data : Array[Array[String]],
) -> Unit {
  if data.length() == 0 {
    abort("数据不能为空")
  }
  let n_samples = data.length()
  let n_features = data[0].length()
  self.categories = Array::make(n_features, [])

  // 对每个特征提取唯一类别
  for i = 0; i < n_features; i = i + 1 {
    let unique_list = []
    for j = 0; j < n_samples; j = j + 1 {
      let mut found = false
      for k = 0; k < unique_list.length(); k = k + 1 {
        if unique_list[k] == data[j][i] {
          found = true
          break
        }
      }
      if not(found) {
        unique_list.push(data[j][i])
      }
    }

    // 转换为固定数组并排序
    let categories_array = Array::make(unique_list.length(), "")
    for k = 0; k < unique_list.length(); k = k + 1 {
      categories_array[k] = unique_list[k]
    }
    categories_array.sort_by(fn(a, b) { compare_strings(a, b) })
    self.categories[i] = categories_array
  }
  self.fitted = true
}

///|
/// 转换数据为独热编码
/// @param data 待转换的数据
/// @return 独热编码后的数据（0.0 或 1.0）
pub fn OneHotEncoder::transform(
  self : OneHotEncoder,
  data : Array[Array[String]],
) -> Array[Array[Double]] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let n_samples = data.length()
  let n_features = data[0].length()

  // 计算总共需要多少列
  let mut total_features = 0
  for i = 0; i < n_features; i = i + 1 {
    total_features = total_features + self.categories[i].length()
  }
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let row = Array::make(total_features, 0.0)
    let mut col_index = 0
    for j = 0; j < n_features; j = j + 1 {
      let value = data[i][j]
      let category_index = find_index(self.categories[j], value)
      if category_index == -1 {
        abort("未知的类别: " + value)
      }
      row[col_index + category_index] = 1.0
      col_index = col_index + self.categories[j].length()
    }
    result[i] = row
  }
  result
}

///|
/// 拟合并转换
pub fn OneHotEncoder::fit_transform(
  self : OneHotEncoder,
  data : Array[Array[String]],
) -> Array[Array[Double]] {
  self.fit(data)
  self.transform(data)
}

///|
/// 获取特征名称（用于理解编码后的列）
pub fn OneHotEncoder::get_feature_names(self : OneHotEncoder) -> Array[String] {
  if not(self.fitted) {
    abort("必须先调用 fit() 方法")
  }
  let feature_list = []
  for i = 0; i < self.categories.length(); i = i + 1 {
    for j = 0; j < self.categories[i].length(); j = j + 1 {
      feature_list.push(
        "feature_" + i.to_string() + "_" + self.categories[i][j],
      )
    }
  }
  let result = Array::make(feature_list.length(), "")
  for i = 0; i < feature_list.length(); i = i + 1 {
    result[i] = feature_list[i]
  }
  result
}

///|
/// 辅助函数：查找元素索引
fn find_index(arr : Array[String], item : String) -> Int {
  for i = 0; i < arr.length(); i = i + 1 {
    if arr[i] == item {
      return i
    }
  }
  -1
}

///|
/// 二值化器 (Binarizer)
/// 根据阈值将数据二值化
pub struct Binarizer {
  threshold : Double
}

///|
pub fn Binarizer::new(threshold? : Double = 0.0) -> Binarizer {
  { threshold, }
}

///|
/// 转换数据（二值化不需要拟合）
pub fn Binarizer::transform(
  self : Binarizer,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  let n_samples = data.length()
  let result = Array::make(n_samples, [])
  for i = 0; i < n_samples; i = i + 1 {
    let n_features = data[i].length()
    let row = Array::make(n_features, 0.0)
    for j = 0; j < n_features; j = j + 1 {
      row[j] = if data[i][j] > self.threshold { 1.0 } else { 0.0 }
    }
    result[i] = row
  }
  result
}

///|
/// 拟合并转换
pub fn Binarizer::fit_transform(
  self : Binarizer,
  data : Array[Array[Double]],
) -> Array[Array[Double]] {
  self.transform(data)
}
