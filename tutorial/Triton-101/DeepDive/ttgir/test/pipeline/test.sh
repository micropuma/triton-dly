triton-opt ./loop-pipeline.mlir \
     -split-input-file \
     -tritongpu-assign-latencies \
     -tritongpu-schedule-loops \
     -tritongpu-pipeline=num-stages=3 \
     -canonicalize \
     -o loop-pipeline-result.mlir

# ========================= Simple Test =========================
triton-opt ./loop-simple1.mlir \
     -split-input-file \
     -tritongpu-assign-latencies \
     -tritongpu-schedule-loops \
     -tritongpu-pipeline=num-stages=3 \
     -canonicalize \
     -o loop-simple1-result.mlir

