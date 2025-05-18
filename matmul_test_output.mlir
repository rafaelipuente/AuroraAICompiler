module {
  func.func @main(%arg0: tensor<1x5xf32>) -> (tensor<1x10xf32>) {
    %cst_W = "dummy.constant"() : () -> tensor<5x10xf32>
    %Y = aurora.matmul(%arg0, %cst_W) : (tensor<1x5xf32>, tensor<5x10xf32>) -> tensor<1x10xf32>
    return %Y : tensor<1x10xf32>
  }
}