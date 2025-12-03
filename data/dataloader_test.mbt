///|
/// DataLoader 行为测试

///|
test "dataloader drop_last and count" {
  let features = [[1.0], [2.0], [3.0], [4.0], [5.0]]
  let labels = [1.0, 2.0, 3.0, 4.0, 5.0]
  let ds = Dataset::new(features, labels)
  let loader = DataLoader::new(ds, batch_size=2, shuffle=false, drop_last=true)
  let batches = loader.all_batches()
  assert_eq(batches.length(), 2)
  assert_eq(batches[0].0.length(), 2)
  assert_eq(batches[1].0.length(), 2)
}

///|
test "dataloader deterministic shuffle" {
  let features = [[1.0], [2.0], [3.0], [4.0]]
  let labels = [10.0, 20.0, 30.0, 40.0]
  let ds = Dataset::new(features, labels)
  let loader_a = DataLoader::new(ds, batch_size=2, shuffle=true, seed=7)
  let loader_b = DataLoader::new(ds, batch_size=2, shuffle=true, seed=7)
  match loader_a.next_batch() {
    Some(batch_a) =>
      match loader_b.next_batch() {
        Some(batch_b) => {
          assert_eq(batch_a.0[0][0], batch_b.0[0][0])
          assert_eq(batch_a.1[1], batch_b.1[1])
        }
        None => abort("loader_b no batch")
      }
    None => abort("loader_a no batch")
  }
}

///|
test "dataloader iter walk through all samples" {
  let features = [[1.0], [2.0], [3.0]]
  let labels = [1.0, 2.0, 3.0]
  let ds = Dataset::new(features, labels)
  let loader = DataLoader::new(ds, batch_size=2, shuffle=false)
  let mut seen = 0
  let iter = loader.iter()
  while true {
    match iter.next() {
      Some((fs, ls)) => {
        seen = seen + fs.length()
        assert_eq(fs.length(), ls.length())
      }
      None => break
    }
  }
  assert_eq(seen, 3)
}
