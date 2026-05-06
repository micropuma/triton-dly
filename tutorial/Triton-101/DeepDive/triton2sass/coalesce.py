import triton  
import triton.language as tl
import torch

import os
from pathlib import Path

TRITON_DEBUG_ROOT = (Path.cwd() / "triton_coalesce_debug").resolve()

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
os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-coalesce,ttg-utility"
os.environ["TRITON_ENABLE_LLVM_DEBUG"] = "1"

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
    ],
    key=['n'],
)
@triton.jit
def triton_kernel_add(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n: tl.constexpr,
):
    offset = tl.program_id(0) * n + tl.arange(0, n)
    a = tl.load(a_ptr + offset)
    b = tl.load(b_ptr + offset)
    out = a+b
    tl.store(out_ptr + offset, out)

a = torch.empty(1024, device="cuda", dtype=torch.float32)
b = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
c = torch.empty(1024, device="cuda", dtype=torch.float32)
grid = lambda meta: [1]
triton_kernel_add[grid](a,b,c,n=512)
