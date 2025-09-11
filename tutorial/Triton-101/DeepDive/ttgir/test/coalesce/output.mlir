#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.splat %arg1 : i32 -> tensor<64x1xi32, #blocked>
    %4 = arith.muli %2, %3 : tensor<64x1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %8 = tt.broadcast %6 : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %9 = tt.broadcast %7 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %10 = ttg.convert_layout %9 : tensor<64x64xi32, #blocked1> -> tensor<64x64xi32, #blocked>
    %11 = tt.addptr %8, %10 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked>
    %13 = tt.addptr %12, %2 : tensor<64x1x!tt.ptr<f32>, #blocked>, tensor<64x1xi32, #blocked>
    %14 = tt.splat %arg3 : i32 -> tensor<1x64xi32, #blocked1>
    %15 = arith.muli %7, %14 : tensor<1x64xi32, #blocked1>
    %16 = tt.broadcast %13 : tensor<64x1x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %17 = tt.broadcast %15 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %18 = ttg.convert_layout %17 : tensor<64x64xi32, #blocked1> -> tensor<64x64xi32, #blocked>
    %19 = tt.addptr %16, %18 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi32, #blocked>
    %20 = ttg.convert_layout %11 : tensor<64x64x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %21 = ttg.convert_layout %cst : tensor<64x64xi1, #blocked> -> tensor<64x64xi1, #blocked2>
    %22 = ttg.convert_layout %cst_0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked2>
    %23 = tt.load %20, %21, %22 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %24 = ttg.convert_layout %23 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked>
    %25 = ttg.convert_layout %19 : tensor<64x64x!tt.ptr<f32>, #blocked> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %26 = ttg.convert_layout %24 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %27 = ttg.convert_layout %cst : tensor<64x64xi1, #blocked> -> tensor<64x64xi1, #blocked3>
    tt.store %25, %26, %27 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

