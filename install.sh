# uninstall first
pip uninstall triton

export DEBUG=1
export LLVM_BUILD_DIR=/home/douliyang/large/triton-workspace/llvm-project/build
export PATH=/home/douliyang/large/triton-workspace/llvm-project/build/bin:$PATH

export LLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm
export MLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir

LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
MLIR_DIR=$LLVM_INCLUDE_DIRS \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
