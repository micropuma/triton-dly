// CHECK-LABEL: @test_broadcast_mix_type_op_pattern
tt.func @test_broadcast_mix_type_op_pattern(%arg0: tensor<128x1xf32>, %arg1: f32, %arg2: tensor<1x128xf32>, %arg3: tensor<128x1xi1>) -> (tensor<128x128xf32>) {
    //  CHECK: %[[sel:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<128x1xi1>, tensor<128x1xf32>
    // CHECK-NEXT: %{{.*}} = tt.broadcast %[[sel]] : tensor<128x1xf32> -> tensor<128x128xf32>
    %broadcast0 = tt.broadcast %arg0 : tensor<128x1xf32> -> tensor<128x128xf32>
    %broadcast1 = tt.splat %arg1 : f32 -> tensor<128x128xf32>
    %cond = tt.broadcast %arg3 : tensor<128x1xi1> -> tensor<128x128xi1>
    %sel = arith.select %cond, %broadcast0, %broadcast1 : tensor<128x128xi1>, tensor<128x128xf32>

    tt.return %sel : tensor<128x128xf32>
}
