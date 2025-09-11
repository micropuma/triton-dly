#loc = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":30:0)
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":30:0), 
                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":30:0), 
                             %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":30:0), 
                             %arg3: i32 {tt.divisibility = 16 : i32} loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":30:0)) attributes {noinline = false} {
    // 获取program id                 pid = tl.program_id(axis=0)
    %0 = tt.get_program_id x : i32 loc(#loc1)

    // BLOCK_SIZE是1024
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %c1024_i32_0 = arith.constant 1024 : i32 loc(#loc2)
    %1 = arith.extsi %0 : i32 to i64 loc(#loc2)
    %2 = arith.extsi %c1024_i32_0 : i32 to i64 loc(#loc2)
    %3 = arith.muli %1, %2 : i64 loc(#loc2)
    %c2147483647_i64 = arith.constant 2147483647 : i64 loc(#loc2)
    %c-2147483648_i64 = arith.constant -2147483648 : i64 loc(#loc2)
    %4 = arith.cmpi sle, %3, %c2147483647_i64 : i64 loc(#loc2)
    %5 = arith.cmpi sge, %3, %c-2147483648_i64 : i64 loc(#loc2)
    %6 = arith.andi %4, %5 : i1 loc(#loc2)

    // 对应triton-lang中的block_start = pid * BLOCK_SIZE
    %7 = arith.muli %0, %c1024_i32_0 : i32 loc(#loc2)
    // 开始着手生成tl.arange(0, BLOCK_SIZE)
    %8 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc3)
    %9 = tt.splat %7 : i32 -> tensor<1024xi32> loc(#loc4)
    %10 = arith.extsi %9 : tensor<1024xi32> to tensor<1024xi64> loc(#loc4)
    %11 = arith.extsi %8 : tensor<1024xi32> to tensor<1024xi64> loc(#loc4)
    // 对应offsets = block_start + tl.arange(0, BLOCK_SIZE)
    %12 = arith.addi %10, %11 : tensor<1024xi64> loc(#loc4)
    %c2147483647_i64_1 = arith.constant 2147483647 : i64 loc(#loc4)
    %c-2147483648_i64_2 = arith.constant -2147483648 : i64 loc(#loc4)
    %cst = arith.constant dense<2147483647> : tensor<1024xi64> loc(#loc4)
    %13 = arith.cmpi sle, %12, %cst : tensor<1024xi64> loc(#loc4)
    %cst_3 = arith.constant dense<-2147483648> : tensor<1024xi64> loc(#loc4)
    %14 = arith.cmpi sge, %12, %cst_3 : tensor<1024xi64> loc(#loc4)
    %15 = arith.andi %13, %14 : tensor<1024xi1> loc(#loc4)
    %16 = arith.addi %9, %8 : tensor<1024xi32> loc(#loc4)

    // 对应mask = offsets < n_elements
    %17 = tt.splat %arg3 : i32 -> tensor<1024xi32> loc(#loc5)
    %18 = arith.cmpi slt, %16, %17 : tensor<1024xi32> loc(#loc5)

    // arg0是指向X存储空间的指针，用一个1024的指针数组，该指针数组里面的ptr每个相应移位
    // 来体现一个block的不同thread的访存
    %19 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc6)
    %20 = tt.addptr %19, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc6)

    // 带mask的访问，将1024个ptr指针指向的内容加载成一个tensor<1024xf32>
    %21 = tt.load %20, %18 : tensor<1024x!tt.ptr<f32>> loc(#loc7)

    %22 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc8)
    %23 = tt.addptr %22, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc8)
    %24 = tt.load %23, %18 : tensor<1024x!tt.ptr<f32>> loc(#loc9)

    %25 = arith.addf %21, %24 : tensor<1024xf32> loc(#loc10)
    %26 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc11)
    %27 = tt.addptr %26, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc11)
    tt.store %27, %25, %18 : tensor<1024x!tt.ptr<f32>> loc(#loc12)
    tt.return loc(#loc13)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":39:24)
#loc2 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":52:24)
#loc3 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":53:41)
#loc4 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":53:28)
#loc5 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":55:21)
#loc6 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":58:24)
#loc7 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":58:16)
#loc8 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":59:24)
#loc9 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":59:16)
#loc10 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":60:17)
#loc11 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":62:26)
#loc12 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":62:35)
#loc13 = loc("/mnt/home/douliyang/triton-workspace/triton-tutorial/tutorial/Triton-101/Debug/vector_add.py":62:4)