import triton
import triton.language as tl
import torch

import os
from pathlib import Path

TRITON_DEBUG_ROOT = (Path.cwd() / "triton_reduction_debug").resolve()

CACHE_DIR = TRITON_DEBUG_ROOT / "cache"
DUMP_DIR = TRITON_DEBUG_ROOT / "kernel_dump"
MLIR_DUMP_FILE = TRITON_DEBUG_ROOT / "mlir_pass_dump" / "mlir_dump.log"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DUMP_DIR.mkdir(parents=True, exist_ok=True)
MLIR_DUMP_FILE.parent.mkdir(parents=True, exist_ok=True)

os.environ["TRITON_CACHE_DIR"] = str(CACHE_DIR)
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["MLIR_DUMP_PATH"] = str(MLIR_DUMP_FILE)
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_DUMP_DIR"] = str(DUMP_DIR)
os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions,ttgpuir,ttg-utility"
os.environ["TRITON_ENABLE_LLVM_DEBUG"] = "1"


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
    ],
    key=["block_n", "block_m"],
)
@triton.jit
def triton_kernel_sum_dim1(
    in_ptr: tl.tensor,
    out_ptr: tl.tensor,
    stride_in_n: tl.constexpr,
    stride_in_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    block_n: tl.constexpr,
    block_m: tl.constexpr,
):
    offsets_n = tl.arange(0, block_n)[:, None]
    offsets_m = tl.arange(0, block_m)[None, :]
    offset_in = stride_in_n * offsets_n + stride_in_m * offsets_m
    offset_out = stride_out_n * tl.arange(0, block_n)

    a = tl.load(in_ptr + offset_in)
    b = tl.sum(a, axis=1)
    tl.store(out_ptr + offset_out, b)


n = 1
m = 32
dtype = torch.float32

a = torch.empty((n, m), dtype=dtype, device="cuda")
b = torch.empty((n,), dtype=dtype, device="cuda")

grid = lambda meta: [1]
triton_kernel_sum_dim1[grid](
    a,
    b,
    a.stride(0),
    a.stride(1),
    b.stride(0),
    block_n=n,
    block_m=m,
)
