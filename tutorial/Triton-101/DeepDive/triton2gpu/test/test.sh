triton-opt ./test.mlir \
           -split-input-file \
           -convert-triton-to-tritongpu='target=cuda:86 num-warps=2' \
           -o output.mlir

triton-opt ./dot-operand.mlir \
           -split-input-file \
           -convert-triton-to-tritongpu='target=cuda:86 num-warps=2' \
           -debug-only=dialect-conversion \
           -o output.mlir \
           2>&1 | tee log.txt