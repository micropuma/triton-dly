// CHECK-LABEL: @test_combine_addptr_pattern
tt.func @test_combine_addptr_pattern(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 10 : i32
    %off1 = arith.constant 15 : i32

    // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<25> : tensor<8xi32>

    %base_ = tt.splat %base : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>      // tt.splat 支持canonicalize正则化

    // CHECK-NEXT: %[[tmp0:.*]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.splat %off0 : i32 -> tensor<8xi32>
    %idx1 = tt.splat %off1 : i32 -> tensor<8xi32>

    // CHECK-NEXT: %1 = tt.addptr %[[tmp0]], %[[cst]] : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>

    tt.return %ptr1 : tensor<8x!tt.ptr<f32>>
}