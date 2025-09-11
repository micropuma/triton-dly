#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @ops() {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #blocked>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #blocked1>
    %cst_1 = arith.constant dense<3.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = ttg.convert_layout %cst : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %1 = ttg.convert_layout %cst_0 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %2 = ttg.convert_layout %cst_1 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked2>
    %3 = tt.dot %0, %1, %2 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x128xf32, #blocked2>
    tt.return
  }
}

