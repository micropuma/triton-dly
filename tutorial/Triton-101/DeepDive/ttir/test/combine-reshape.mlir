// CHECK-LABEL: @test_canonicalize_reshape
tt.func @test_canonicalize_reshape(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> (tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>) {
    %reshape0 = tt.reshape %arg0 : tensor<8xf32> -> tensor<2x4xf32>
    // CHECK: %{{.*}} = tt.reshape %arg0 : tensor<8xf32> -> tensor<4x2xf32>
    %reshape1 = tt.reshape %reshape0 : tensor<2x4xf32> -> tensor<4x2xf32>

    %splat = tt.splat %arg1 : tensor<f32> -> tensor<8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : tensor<f32> -> tensor<2x2x2xf32>
    %reshape2 = tt.reshape %splat : tensor<8xf32> -> tensor<2x2x2xf32>

    %reshape3 = tt.reshape %arg0 : tensor<8xf32> -> tensor<8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<8xf32>
    %add = arith.addf %reshape3, %arg0 : tensor<8xf32>

    // CHECK: %{{.*}} = tt.reshape %arg0 allow_reorder : tensor<8xf32> -> tensor<2x2x2xf32>
    %view = tt.reshape %reshape0 allow_reorder : tensor<2x4xf32> -> tensor<2x2x2xf32>

    tt.return %reshape1, %reshape2, %add, %view : tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>
}