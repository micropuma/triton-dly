# uninstall first
pip uninstall triton

export DEBUG=1
export LLVM_DIR=/home/douliyang/large/triton-workspace/llvm-project/build

LLVM_SYSPATH=$LLVM_DIR \
LLVM_INCLUDE_DIRS=$LLVM_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_DIR/lib \
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
