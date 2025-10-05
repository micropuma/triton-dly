#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 2x128x32>
    %14 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %15 = ttg.async_copy_global_to_local %4, %13 mask %14 : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable, 2x128x32>
    %16 = ttg.async_commit_group tokens %15
    %17 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 2x32x128>
    %18 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %19 = ttg.async_copy_global_to_local %9, %17 mask %18 other %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable, 2x32x128>
    %20 = ttg.async_commit_group tokens %19
    %21 = arith.addi %arg0, %arg2 : index
    %22 = arith.cmpi slt, %21, %arg1 : index
    %23 = tt.addptr %4, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %24 = tt.addptr %9, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %25 = ttg.memdesc_index %10[%c1_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 2x128x32>
    %26 = tt.splat %22 : i1 -> tensor<128x32xi1, #blocked1>
    %27 = ttg.async_copy_global_to_local %23, %25 mask %26 : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable, 2x128x32>
    %28 = ttg.async_commit_group tokens %27
    %29 = ttg.memdesc_index %11[%c1_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 2x32x128>
    %30 = tt.splat %22 : i1 -> tensor<32x128xi1, #blocked>
    %31 = ttg.async_copy_global_to_local %24, %29 mask %30 other %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable, 2x32x128>
    %32 = ttg.async_commit_group tokens %31
    %33:9 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %23, %arg7 = %24, %arg8 = %cst_2, %arg9 = %c1_i32, %arg10 = %c-1_i32, %arg11 = %16, %arg12 = %28, %arg13 = %20, %arg14 = %32) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %35 = arith.muli %arg2, %c2 : index
      %36 = arith.subi %arg1, %35 : index
      %37 = arith.cmpi slt, %arg5, %36 : index
      %38 = arith.addi %arg10, %c1_i32 : i32
      %39 = arith.cmpi sge, %38, %c2_i32 : i32
      %40 = arith.select %39, %c0_i32, %38 : i32
      %41 = ttg.async_wait %arg11, %arg13 {num = 2 : i32}
      %42 = ttg.memdesc_index %10[%40] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 2x128x32>
      %43 = ttg.local_load %42 token %41 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 2x128x32> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %44 = ttg.memdesc_index %11[%40] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 2x32x128>
      %45 = ttg.local_load %44 token %41 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 2x32x128> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %46 = arith.mulf %45, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %47 = tt.dot %43, %46, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %48 = tt.addptr %arg6, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %49 = tt.addptr %arg7, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %50 = arith.addi %arg9, %c1_i32 : i32
      %51 = arith.cmpi sge, %50, %c2_i32 : i32
      %52 = arith.select %51, %c0_i32, %50 : i32
      %53 = ttg.memdesc_index %10[%52] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 2x128x32>
      %54 = tt.splat %37 : i1 -> tensor<128x32xi1, #blocked1>
      %55 = ttg.async_copy_global_to_local %48, %53 mask %54 : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable, 2x128x32>
      %56 = ttg.async_commit_group tokens %55
      %57 = ttg.memdesc_index %11[%52] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable, 2x32x128>
      %58 = tt.splat %37 : i1 -> tensor<32x128xi1, #blocked>
      %59 = ttg.async_copy_global_to_local %49, %57 mask %58 other %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable, 2x32x128>
      %60 = ttg.async_commit_group tokens %59
      scf.yield %48, %49, %47, %52, %40, %arg12, %56, %arg14, %60 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %34 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %11 : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    ttg.local_dealloc %10 : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    tt.return %33#2 : tensor<128x128xf32, #mma>
  }
}

