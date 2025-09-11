module {
  tt.func public @rewrite_load(%arg0: !tt.ptr<f16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
    %0 = arith.extsi %c0_i32 : i32 to i64
    %1 = arith.extsi %c0_i32 : i32 to i64
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
    %3 = tt.splat %0 : i64 -> tensor<128xi64>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = arith.extsi %4 : tensor<128xi32> to tensor<128xi64>
    %6 = arith.addi %3, %5 : tensor<128xi64>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
    %8 = tt.splat %c1_i64 : i64 -> tensor<128x1xi64>
    %9 = arith.muli %7, %8 : tensor<128x1xi64>
    %10 = tt.broadcast %9 : tensor<128x1xi64> -> tensor<128x32xi64>
    %11 = tt.addptr %2, %10 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
    %12 = tt.splat %1 : i64 -> tensor<32xi64>
    %13 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %14 = arith.extsi %13 : tensor<32xi32> to tensor<32xi64>
    %15 = arith.addi %12, %14 : tensor<32xi64>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
    %17 = tt.splat %c1_i64 : i64 -> tensor<1x32xi64>
    %18 = arith.muli %16, %17 : tensor<1x32xi64>
    %19 = tt.broadcast %18 : tensor<1x32xi64> -> tensor<128x32xi64>
    %20 = tt.addptr %11, %19 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
    %c0_i64 = arith.constant 0 : i64
    %21 = tt.splat %c0_i64 : i64 -> tensor<1x32xi64>
    %22 = arith.cmpi sge, %16, %21 : tensor<1x32xi64>
    %23 = tt.splat %c32_i64 : i64 -> tensor<1x32xi64>
    %24 = arith.cmpi slt, %16, %23 : tensor<1x32xi64>
    %25 = arith.andi %22, %24 : tensor<1x32xi1>
    %26 = tt.broadcast %25 : tensor<1x32xi1> -> tensor<128x32xi1>
    %cst_0 = arith.constant 0x7E00 : f16
    %27 = tt.splat %cst_0 : f16 -> tensor<128x32xf16>
    %28 = tt.load %20, %26, %27 : tensor<128x32x!tt.ptr<f16>>
    tt.return
  }
}

// -----
module {
  tt.func public @rewrite_store(%arg0: !tt.ptr<f16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
    %0 = arith.extsi %c0_i32 : i32 to i64
    %1 = arith.extsi %c0_i32 : i32 to i64
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
    %3 = tt.splat %0 : i64 -> tensor<128xi64>
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = arith.extsi %4 : tensor<128xi32> to tensor<128xi64>
    %6 = arith.addi %3, %5 : tensor<128xi64>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
    %8 = tt.splat %c1_i64 : i64 -> tensor<128x1xi64>
    %9 = arith.muli %7, %8 : tensor<128x1xi64>
    %10 = tt.broadcast %9 : tensor<128x1xi64> -> tensor<128x32xi64>
    %11 = tt.addptr %2, %10 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
    %12 = tt.splat %1 : i64 -> tensor<32xi64>
    %13 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %14 = arith.extsi %13 : tensor<32xi32> to tensor<32xi64>
    %15 = arith.addi %12, %14 : tensor<32xi64>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
    %17 = tt.splat %c1_i64 : i64 -> tensor<1x32xi64>
    %18 = arith.muli %16, %17 : tensor<1x32xi64>
    %19 = tt.broadcast %18 : tensor<1x32xi64> -> tensor<128x32xi64>
    %20 = tt.addptr %11, %19 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
    tt.store %20, %cst : tensor<128x32x!tt.ptr<f16>>
    tt.return
  }
}

// -----
module {
  tt.func public @rewrite_for(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
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
      %14 = tt.splat %arg5 : i64 -> tensor<32xi64>
      %15 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %16 = arith.extsi %15 : tensor<32xi32> to tensor<32xi64>
      %17 = arith.addi %14, %16 : tensor<32xi64>
      %18 = tt.expand_dims %17 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
      %19 = tt.splat %c1_i64 : i64 -> tensor<1x32xi64>
      %20 = arith.muli %18, %19 : tensor<1x32xi64>
      %21 = tt.broadcast %20 : tensor<1x32xi64> -> tensor<128x32xi64>
      %22 = tt.addptr %13, %21 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
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
    } {tt.num_stages = 3 : i32}
    %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
    tt.store %3, %2#0 : tensor<128x32x!tt.ptr<f16>>
    tt.return
  }
}

// -----
module {
  tt.func public @rewrite_if(%arg0: !tt.ptr<f16>, %arg1: i1, %arg2: tensor<128x32xf32>) -> tensor<128x32xf16> {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %0 = arith.extsi %c0_i32 : i32 to i64
    %1 = arith.extsi %c0_i32 : i32 to i64
    %2:3 = scf.if %arg1 -> (tensor<128x32xf16>, i64, i64) {
      %31 = arith.extsi %c32_i32 : i32 to i64
      %32 = arith.addi %0, %31 : i64
      %33 = arith.extsi %c0_i32 : i32 to i64
      %34 = arith.addi %1, %33 : i64
      %35 = arith.truncf %arg2 : tensor<128x32xf32> to tensor<128x32xf16>
      scf.yield %35, %32, %34 : tensor<128x32xf16>, i64, i64
    } else {
      %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
      scf.yield %cst_0, %0, %1 : tensor<128x32xf16>, i64, i64
    }
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
    %4 = tt.splat %2#1 : i64 -> tensor<128xi64>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %6 = arith.extsi %5 : tensor<128xi32> to tensor<128xi64>
    %7 = arith.addi %4, %6 : tensor<128xi64>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
    %9 = tt.splat %c1_i64 : i64 -> tensor<128x1xi64>
    %10 = arith.muli %8, %9 : tensor<128x1xi64>
    %11 = tt.broadcast %10 : tensor<128x1xi64> -> tensor<128x32xi64>
    %12 = tt.addptr %3, %11 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
    %13 = tt.splat %2#2 : i64 -> tensor<32xi64>
    %14 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %15 = arith.extsi %14 : tensor<32xi32> to tensor<32xi64>
    %16 = arith.addi %13, %15 : tensor<32xi64>
    %17 = tt.expand_dims %16 {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>
    %18 = tt.splat %c1_i64 : i64 -> tensor<1x32xi64>
    %19 = arith.muli %17, %18 : tensor<1x32xi64>
    %20 = tt.broadcast %19 : tensor<1x32xi64> -> tensor<128x32xi64>
    %21 = tt.addptr %12, %20 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi64>
    %c0_i64 = arith.constant 0 : i64
    %22 = tt.splat %c0_i64 : i64 -> tensor<1x32xi64>
    %23 = arith.cmpi sge, %17, %22 : tensor<1x32xi64>
    %24 = tt.splat %c32_i64 : i64 -> tensor<1x32xi64>
    %25 = arith.cmpi slt, %17, %24 : tensor<1x32xi64>
    %26 = arith.andi %23, %25 : tensor<1x32xi1>
    %27 = tt.broadcast %26 : tensor<1x32xi1> -> tensor<128x32xi1>
    %cst = arith.constant 0x7E00 : f16
    %28 = tt.splat %cst : f16 -> tensor<128x32xf16>
    %29 = tt.load %21, %27, %28 : tensor<128x32x!tt.ptr<f16>>
    %30 = arith.addf %2#0, %29 : tensor<128x32xf16>
    tt.return %30 : tensor<128x32xf16>
  }
}

// -----
module {
  tt.func public @asm_in_loop(%arg0: !tt.ptr<bf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = arith.extsi %c0_i32 : i32 to i64
    %2 = arith.extsi %c0_i32 : i32 to i64
    %3:2 = scf.for %arg1 = %c0_i32 to %c1_i32 step %c1_i32 iter_args(%arg2 = %1, %arg3 = %2) -> (i64, i64)  : i32 {
      %4:2 = tt.elementwise_inline_asm "asm_multiple_results" {constraints = "=r,=r,r", packed_element = 1 : i32, pure = true} %0 : tensor<16xi32> -> tensor<16xi16>, tensor<16xi16>
      %5 = arith.extsi %c0_i32 : i32 to i64
      %6 = arith.addi %arg2, %5 : i64
      %7 = arith.extsi %c0_i32 : i32 to i64
      %8 = arith.addi %arg3, %7 : i64
      scf.yield %6, %8 : i64, i64
    }
    tt.return
  }
}

