triton-opt ./coalesce.mlir \
    -split-input-file \
    -tritongpu-coalesce \
    -debug \
    -o output.mlir \
    2>&1 | tee log.txt
