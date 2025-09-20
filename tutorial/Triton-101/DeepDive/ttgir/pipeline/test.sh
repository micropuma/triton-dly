# E2E test and process dump
MLIR_ENABLE_DUMP=1 python add.py &> ir_dump.log

# Step by step, from ttir to llir
triton-opt add_kernel.ttir -convert-triton-to-tritongpu='target=cuda:86' &> add.ttgir
triton-opt add.ttgir -test-print-alignment &> add-AxisInfo.ttgir
triton-opt add.ttgir -tritongpu-coalesce &> add-opt.ttgir
triton-opt add-opt.ttgir -allocate-shared-memory-nv -convert-triton-gpu-to-llvm &> add.llir