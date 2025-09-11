#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #ttg.mma<{versionMajor = 2, warpsPerCTA = [2, 2]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
    func @matmul_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
    %a:tensor<32x32xf16, #shared>, %b:tensor<32x32xf16, #shared>) {
        %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
        // CHECK: ldmatrix.sync.aligned.m8n8.x4.shared.b16
        %a_mat = triton_gpu.convert_layout %a : (tensor<32x32xf16, #shared>) -> tensor<32x32xf16, #dot_operand_a>
        %b_mat = triton_gpu.convert_layout %b : (tensor<32x32xf16, #shared>) -> tensor<32x32xf16, #dot_operand_b>

        %28 = tt.dot %a_mat, %b_mat, %cst {allowTF32 = true, transA = false, transB = false} : tensor<32x32xf16, #dot_operand_a> * tensor<32x32xf16, #dot_operand_b> -> tensor<32x32xf32, #mma>
        return
    }
}