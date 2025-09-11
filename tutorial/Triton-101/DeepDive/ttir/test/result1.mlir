module {
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
    %0 = arith.extsi %c0_i32 : i32 to i64
    %1 = arith.extsi %c0_i32 : i32 to i64
    %2:3 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst, %arg4 = %0, %arg5 = %1) -> (tensor<128x32xf16>, i64, i64) {
      // dim0
      %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
      %5 = tt.splat %arg4 : i64 -> tensor<128xi64>
      %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %7 = arith.extsi %6 : tensor<128xi32> to tensor<128xi64>
      %8 = arith.addi %5, %7 : tensor<128xi64>
      %9 = tt.expand_dims %8 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
      %10 = tt.splat %c1_i64 : i64 -> tensor<128x1xi64>
      %11 = arith.muli %9, %10 : tensor<128x1xi64>
      %12 = tt.broadcast %11 : tensor<128x1xi64> -> tensor<128x32xi64>
      %13 = tt.addptr %4, %12 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
      // dim1
      %14 = tt.splat %arg5 : i64 -> tensor<32xi64>
      %15 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %16 = arith.extsi %15 : tensor<32xi32> to tensor<32xi64>
      %17 = arith.addi %14, %16 : tensor<32xi64>
      %18 = tt.expand_dims %17 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
      %19 = tt.splat %c1_i64 : i64 -> tensor<1x32xi64>
      %20 = arith.muli %18, %19 : tensor<1x32xi64>
      %21 = tt.broadcast %20 : tensor<1x32xi64> -> tensor<128x32xi64>
      %22 = tt.addptr %13, %21 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
      // mask
      %c0_i64 = arith.constant 0 : i64
      %23 = tt.splat %c0_i64 : i64 -> tensor<1x32xi64>
      %24 = arith.cmpi sge, %18, %23 : tensor<1x32xi64>
      %25 = tt.splat %c32_i64 : i64 -> tensor<1x32xi64>
      %26 = arith.cmpi slt, %18, %25 : tensor<1x32xi64>
      %27 = arith.andi %24, %26 : tensor<1x32xi1>
      %28 = tt.broadcast %27 : tensor<1x32xi1> -> tensor<128x32xi1>

      %cst_0 = arith.constant 0x7E00 : f16
      %29 = tt.splat %cst_0 : f16 -> tensor<128x32xf16>
      %30 = tt.load %22, %28, %29 : tensor<128x32x!tt.ptr<f16>>
      %31 = arith.addf %arg3, %30 : tensor<128x32xf16>
      %32 = arith.extsi %c32_i32 : i32 to i64
      %33 = arith.addi %arg4, %32 : i64
      %34 = arith.extsi %c0_i32 : i32 to i64
      %35 = arith.addi %arg5, %34 : i64
      scf.yield %31, %33, %35 : tensor<128x32xf16>, i64, i64
    }
    %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
    tt.store %3, %2#0 : tensor<128x32x!tt.ptr<f16>>
    tt.return
  }
}

