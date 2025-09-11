// CHECK-LABEL: @test_canonicalize_broadcast
tt.func @test_canonicalize_broadcast(%arg0: tensor<1x1x8xf32>, %arg1: tensor<f32>) -> (tensor<4x2x8xf32>, tensor<8x8xf32>, tensor<1x1x8xf32>) {
    %broadcast0 = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<1x2x8xf32>
    // CHECK: %{{.*}} = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<4x2x8xf32>
    %broadcast1 = tt.broadcast %broadcast0 : tensor<1x2x8xf32> -> tensor<4x2x8xf32>

    %splat = tt.splat %arg1 : tensor<f32> -> tensor<1x8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : tensor<f32> -> tensor<8x8xf32>
    %broadcast2 = tt.broadcast %splat : tensor<1x8xf32> -> tensor<8x8xf32>

    %broadcast3 = tt.broadcast %arg0 : tensor<1x1x8xf32> -> tensor<1x1x8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<1x1x8xf32>
    %add = arith.addf %broadcast3, %arg0 : tensor<1x1x8xf32>

    tt.return %broadcast1, %broadcast2, %add : tensor<4x2x8xf32>, tensor<8x8xf32>, tensor<1x1x8xf32>
}