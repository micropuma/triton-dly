tt.func @test_combine_broadcast_mul_reducesum(%lhs: tensor<128x64xf32>, %rhs: tensor<64x128xf32>) -> (tensor<128x128xf32>) {
    %expand_lhs = tt.expand_dims %lhs {axis = 2 : i32} : tensor<128x64xf32> -> tensor<128x64x1xf32>
    %expand_rhs = tt.expand_dims %rhs {axis = 0 : i32} : tensor<64x128xf32> -> tensor<1x64x128xf32>
    %a = tt.broadcast %expand_lhs : tensor<128x64x1xf32> -> tensor<128x64x128xf32>
    %b = tt.broadcast %expand_rhs : tensor<1x64x128xf32> -> tensor<128x64x128xf32>
    %mul = arith.mulf %a, %b : tensor<128x64x128xf32>
    %reduce = "tt.reduce" (%mul) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %add : f32
    }) {axis = 1 : i32} : (tensor<128x64x128xf32>) -> tensor<128x128xf32>
    tt.return %reduce : tensor<128x128xf32>
}