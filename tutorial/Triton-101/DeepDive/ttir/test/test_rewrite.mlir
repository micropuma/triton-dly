tt.func public @rewrite_tensor_ptr(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x32xf16>>
  %1:2 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst, %arg4 = %0) -> (tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>) {
    %3 = tt.load %arg4 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
    %4 = arith.addf %arg3, %3 : tensor<128x32xf16>
    %5 = tt.advance %arg4, [%c32_i32, %c0_i32] : <tensor<128x32xf16>>
    scf.yield %4, %5 : tensor<128x32xf16>, !tt.ptr<tensor<128x32xf16>>
  }
  %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
  tt.store %2, %1#0 : tensor<128x32x!tt.ptr<f16>>
  tt.return
}
