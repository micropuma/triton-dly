# <source-dir> should be the local checkout directory for
#   https://github.com/llvm/llvm-project/tree/main/llvm
# <target-dir> is where to put the compiled LLVM/MLIR artifacts
triton-configure-mlir() {
  if (( $# < 3 ))
  then
    echo "usage: $0 <source-dir> <target-dir> <build-type>"
    return 1
  fi

  SOURCE_DIR=$1; shift
  TARGET_DIR=$1; shift
  BUILD_TYPE=$1; shift

  cmake -GNinja \
    -S ${SOURCE_DIR} -B ${TARGET_DIR} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=$(which clang) -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir" \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86;AArch64"
}

# <source-dir> should be the local checkout directory for
#   https://github.com/triton-lang/triton
# <target-dir> is where to put the compiled Triton artifacts
# <mlir-dir> should be the LLVM/MLIR artifacts directory
triton-cmake() {
  if (( $# < 4 ))
  then
    echo "usage: $0 <source-dir> <target-dir> <build-type> <mlir-dir>"
    return 1
  fi

  SOURCE_DIR=$1; shift
  TARGET_DIR=$1; shift
  BUILD_TYPE=$1; shift
  MLIR_DIR=$1;   shift

  if [[ "$(uname)" == "Darwin" ]]; then
    LINKER_FLAGS=()
  else
    LINKER_FLAGS=(
      "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld"
      "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld"
      "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld"
    )
  fi

  REPO_BASE_DIR=$(git rev-parse --show-toplevel)

  cmake -GNinja \
    -S ${SOURCE_DIR} -B ${TARGET_DIR} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DTRITON_CODEGEN_BACKENDS="amd;nvidia" \
    -DLLVM_INCLUDE_DIRS=${MLIR_DIR}/include \
    -DLLVM_LIBRARY_DIR=${MLIR_DIR}/lib \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER=lld ${LINKER_FLAGS[@]} \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTRITON_BUILD_PYTHON_MODULE=ON \
    -DTRITON_BUILD_PROTON=ON \
    -DCUPTI_INCLUDE_DIR=${REPO_BASE_DIR}/third_party/nvidia/backend/include \
    -DROCTRACER_INCLUDE_DIR=${REPO_BASE_DIR}/third_party/amd/backend/include \
    -DJSON_INCLUDE_DIR=$HOME/.triton/json/include
}

triton-configure-mlir llvm-project/llvm build/mlir-debug Debug
cmake --build build/mlir-debug

