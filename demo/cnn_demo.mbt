///|
/// 简单 CNN 训练示例（单步）

///|
pub fn cnn_train_demo() -> Double {
  let conv = @nn.Conv2dModule::new(1, 1, 1, stride=1, padding=0, seed=1)
  let bn = @nn.BatchNorm2d::new(1)
  let maxp = @nn.MaxPool2dModule::new(2)
  let flat = @nn.Flatten2dModule::new()
  let fc = @nn.LinearModule::new(1, 2, seed=2)
  let seq = @nn.Sequential::new([
    @nn.module_conv2d(conv),
    @nn.module_batchnorm2d(bn),
    @nn.module_relu(),
    @nn.module_maxpool2d(maxp),
    @nn.module_flatten2d(flat),
    @nn.module_linear(fc),
  ])
  let x = @nn.AutoTensor::new(
    @math.Tensor::new([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
  )
  let labels = [0]
  let (logits, ctx) = seq.forward(x)
  let loss = logits.softmax_cross_entropy(labels)
  loss.backward()
  seq.sync_grads(ctx)
  let params = seq.parameters()
  let opt = @nn.SGDParams::new(0.1, params)
  opt.step(params)
  loss.value().get([0])
}
