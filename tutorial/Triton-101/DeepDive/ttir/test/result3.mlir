module {
  tt.func @test_splat_elementwise_pattern(%arg0: f32) -> (tensor<128x128xf32>, tensor<128x128x!tt.ptr<f32>>) {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %0 = arith.addf %arg0, %cst : f32
    %1 = tt.splat %0 : f32 -> tensor<128x128xf32>
    %2 = tt.int_to_ptr %c1_i64 : i64 -> !tt.ptr<f32>
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    tt.return %1, %3 : tensor<128x128xf32>, tensor<128x128x!tt.ptr<f32>>
  }
  tt.func @test_broadcast_elementwise_pattern(%arg0: tensor<128x1xf32>) -> (tensor<128x128xf32>, tensor<128x32xf32>) {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x1xf32>
    %0 = math.absf %arg0 : tensor<128x1xf32>
    %1 = tt.broadcast %0 : tensor<128x1xf32> -> tensor<128x128xf32>
    %2 = arith.addf %arg0, %cst : tensor<128x1xf32>
    %3 = tt.broadcast %2 : tensor<128x1xf32> -> tensor<128x32xf32>
    tt.return %1, %3 : tensor<128x128xf32>, tensor<128x32xf32>
  }
  tt.func @test_broadcast_binary_op_pattern(%arg0: tensor<128x1xf32>, %arg1: tensor<128x1xf32>, %arg2: tensor<1x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %0 = tt.broadcast %arg0 : tensor<128x1xf32> -> tensor<128x128xf32>
    %1 = arith.mulf %arg0, %arg1 : tensor<128x1xf32>
    %2 = tt.broadcast %1 : tensor<128x1xf32> -> tensor<128x128xf32>
    %3 = tt.broadcast %arg2 : tensor<1x128xf32> -> tensor<128x128xf32>
    %4 = arith.mulf %0, %3 : tensor<128x128xf32>
    tt.return %2, %4 : tensor<128x128xf32>, tensor<128x128xf32>
  }
  tt.func @test_broadcast_mix_type_op_pattern(%arg0: tensor<128x1xf32>, %arg1: f32, %arg2: tensor<1x128xf32>, %arg3: tensor<128x1xi1>) -> tensor<128x128xf32> {
    %0 = tt.splat %arg1 : f32 -> tensor<128x1xf32>
    %1 = arith.select %arg3, %arg0, %0 : tensor<128x1xi1>, tensor<128x1xf32>
    %2 = tt.broadcast %1 : tensor<128x1xf32> -> tensor<128x128xf32>
    tt.return %2 : tensor<128x128xf32>
  }
}

