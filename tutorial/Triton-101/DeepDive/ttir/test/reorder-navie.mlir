// CHECK-LABEL: @test_broadcast_binary_op_pattern
tt.func @test_broadcast_binary_op_pattern(%arg0: tensor<128x1xf32>, %arg1: tensor<128x1xf32>, %arg2: tensor<1x128xf32>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    // CHECK: %[[mul:.*]] = arith.mulf %{{.*}}, %{{.*}} : tensor<128x1xf32>
    // CHECK-NEXT: %{{.*}} = tt.broadcast %[[mul]] : tensor<128x1xf32> -> tensor<128x128xf32>
    %broadcast0 = tt.broadcast %arg0 : tensor<128x1xf32> -> tensor<128x128xf32>
    %broadcast1 = tt.broadcast %arg1 : tensor<128x1xf32> -> tensor<128x128xf32>
    %mul = arith.mulf %broadcast0, %broadcast1 : tensor<128x128xf32>

    // CHECK: %[[mul:.*]] = arith.mulf %{{.*}}, %{{.*}} : tensor<128x128xf32>
    %broadcast2 = tt.broadcast %arg2 : tensor<1x128xf32> -> tensor<128x128xf32>
    %mul1 = arith.mulf %broadcast0, %broadcast2 : tensor<128x128xf32>

    tt.return %mul, %mul1 : tensor<128x128xf32>, tensor<128x128xf32>
}
