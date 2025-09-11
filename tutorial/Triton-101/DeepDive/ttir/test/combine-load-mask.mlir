// CHECK-LABEL: @test_canonicalize_masked_load_pattern
tt.func @test_canonicalize_masked_load_pattern(%ptr: tensor<8x!tt.ptr<f32>>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %true_mask = arith.constant dense<true> : tensor<8xi1>
    %false_mask = arith.constant dense<false> : tensor<8xi1>
    %other_val = arith.constant dense<0.0> : tensor<8xf32>

    // true_mask with other
    // CHECK: %[[res1:.*]] = tt.load %{{.*}} : tensor<8x!tt.ptr<f32>>
    %x = tt.load %ptr, %true_mask : tensor<8x!tt.ptr<f32>>

    // true_mask without other
    // CHECK: %[[res2:.*]] = tt.load %{{.*}} : tensor<8x!tt.ptr<f32>>
    %y = tt.load %ptr, %true_mask, %other_val : tensor<8x!tt.ptr<f32>>

    // false_mask with other. It should become "other" (i.e., %y)
    %z = tt.load %ptr, %false_mask, %y : tensor<8x!tt.ptr<f32>>

    // CHECK: tt.return %[[res1]], %[[res2]], %[[res2]] : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
    tt.return %x, %y, %z: tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}