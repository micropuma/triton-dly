module {
  tt.func @test_combine_dot_add_pattern() -> tensor<128x128xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    %0 = tt.dot %cst, %cst_0, %cst_1 : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    tt.return %0 : tensor<128x128xf32>
  }
  tt.func @test_combine_dot_add_rev_pattern() -> tensor<128x128xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    %cst_1 = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    %0 = tt.dot %cst, %cst_0, %cst_1 : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    tt.return %0 : tensor<128x128xf32>
  }
  tt.func @test_combine_addptr_pattern(%arg0: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %cst = arith.constant dense<25> : tensor<8xi32>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %1 = tt.addptr %0, %cst : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.return %1 : tensor<8x!tt.ptr<f32>>
  }
  tt.func @test_combine_addptr_pattern_discardableattrs(%arg0: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %c12_i32 = arith.constant 12 : i32
    %0 = tt.addptr %arg0, %c12_i32 {tt.constancy = 8 : i32, tt.contiguity = 512 : i32, tt.divisibility = 16 : i32} : !tt.ptr<f32>, i32
    tt.return %0 : !tt.ptr<f32>
  }
  tt.func @test_combine_addptr_pattern_discardableattrs_disallowed(%arg0: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %c12_i32 = arith.constant 12 : i32
    %0 = tt.addptr %arg0, %c12_i32 {tt.divisibility = 16 : i32} : !tt.ptr<f32>, i32
    tt.return %0 : !tt.ptr<f32>
  }
  tt.func @test_combine_addptr_pattern_i64(%arg0: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %cst = arith.constant dense<25> : tensor<8xi64>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %1 = tt.addptr %0, %cst : tensor<8x!tt.ptr<f32>>, tensor<8xi64>
    tt.return %1 : tensor<8x!tt.ptr<f32>>
  }
  tt.func @test_combine_addptr_pattern_scalar(%arg0: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %c25_i32 = arith.constant 25 : i32
    %0 = tt.addptr %arg0, %c25_i32 : !tt.ptr<f32>, i32
    tt.return %0 : !tt.ptr<f32>
  }
  tt.func @test_not_combine_addptr_pattern_1(%arg0: !tt.ptr<f32>, %arg1: tensor<8xi32>) -> tensor<8x!tt.ptr<f32>> {
    %cst = arith.constant dense<15> : tensor<8xi32>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %1 = tt.addptr %0, %arg1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %2 = tt.addptr %1, %cst : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.return %2 : tensor<8x!tt.ptr<f32>>
  }
  tt.func @test_not_combine_addptr_pattern(%arg0: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %cst = arith.constant dense<15> : tensor<8xi32>
    %cst_0 = arith.constant dense<10> : tensor<8xi16>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %1 = tt.addptr %0, %cst_0 : tensor<8x!tt.ptr<f32>>, tensor<8xi16>
    %2 = tt.addptr %1, %cst : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.return %2 : tensor<8x!tt.ptr<f32>>
  }
  tt.func @test_combine_select_masked_load_pattern(%arg0: tensor<8x!tt.ptr<f32>>, %arg1: i1) -> (tensor<8xf32>, tensor<8xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %0 = tt.splat %arg1 : i1 -> tensor<8xi1>
    %1 = tt.load %arg0, %0, %cst : tensor<8x!tt.ptr<f32>>
    %2 = tt.load %arg0, %0, %cst : tensor<8x!tt.ptr<f32>>
    tt.return %1, %2 : tensor<8xf32>, tensor<8xf32>
  }
  tt.func @test_combine_select_masked_load_fail_pattern(%arg0: tensor<8x!tt.ptr<f32>>, %arg1: tensor<8xf32>, %arg2: tensor<8xi1>, %arg3: i1, %arg4: i1) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %0 = arith.select %arg3, %arg1, %cst : tensor<8xf32>
    %1 = tt.load %arg0, %arg2, %cst : tensor<8x!tt.ptr<f32>>
    %2 = arith.select %arg3, %1, %cst : tensor<8xf32>
    %3 = tt.splat %arg3 : i1 -> tensor<8xi1>
    %4 = tt.load %arg0, %3, %cst : tensor<8x!tt.ptr<f32>>
    %5 = arith.select %arg4, %4, %cst : tensor<8xf32>
    tt.return %0, %2, %5 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
  }
  tt.func @test_canonicalize_masked_load_pattern(%arg0: tensor<8x!tt.ptr<f32>>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %0 = tt.load %arg0 : tensor<8x!tt.ptr<f32>>
    %1 = tt.load %arg0 : tensor<8x!tt.ptr<f32>>
    tt.return %0, %1, %1 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
  }
  tt.func @test_canonicalize_masked_store_pattern(%arg0: tensor<8x!tt.ptr<f32>>, %arg1: tensor<8xf32>) {
    tt.store %arg0, %arg1 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
  tt.func @test_canonicalize_masked_store_fail_pattern(%arg0: tensor<8x!tt.ptr<f32>>, %arg1: tensor<8xf32>, %arg2: tensor<8xi1>) {
    tt.store %arg0, %arg1, %arg2 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
  tt.func @test_canonicalize_expand_dims(%arg0: tensor<f32>, %arg1: tensor<1xf32>) -> (tensor<1x8xf32>, tensor<8x8xf32>) {
    %0 = tt.splat %arg0 : tensor<f32> -> tensor<1x8xf32>
    %1 = tt.expand_dims %arg1 {axis = 0 : i32} : tensor<1xf32> -> tensor<1x1xf32>
    %2 = tt.broadcast %1 : tensor<1x1xf32> -> tensor<8x8xf32>
    tt.return %0, %2 : tensor<1x8xf32>, tensor<8x8xf32>
  }
  tt.func @test_canonicalize_view(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> (tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>) {
    %0 = tt.reshape %arg0 allow_reorder : tensor<8xf32> -> tensor<4x2xf32>
    %1 = tt.splat %arg1 : tensor<f32> -> tensor<2x2x2xf32>
    %2 = arith.addf %arg0, %arg0 : tensor<8xf32>
    %3 = tt.reshape %arg0 allow_reorder : tensor<8xf32> -> tensor<2x2x2xf32>
    tt.return %0, %1, %2, %3 : tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>, tensor<2x2x2xf32>
  }
}

