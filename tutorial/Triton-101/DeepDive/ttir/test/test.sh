triton-opt ./rewrite_tensor_ptr.mlir -triton-rewrite-tensor-pointer -split-input-file 2>&1 | tee result.mlir
triton-opt ./test_rewrite.mlir -triton-rewrite-tensor-pointer -split-input-file 2>&1 | tee result1.mlir

triton-opt ./combine.mlir -canonicalize -triton-combine | FileCheck ./combine.mlir
triton-opt ./combine1.mlir -canonicalize -triton-combine 2>&1 | tee result2.mlir

triton-opt ./reorder.mlir -triton-reorder-broadcast 2>&1 | tee result3.mlir
