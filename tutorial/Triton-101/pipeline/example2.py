import torch

import triton
import triton.language as tl
from triton.tools.disasm import get_sass


import triton
import triton.language as tl
import torch

@triton.jit
def block_kernel(x_ptr, o_ptr):
    absurd_shape: tl.constexpr = (2, -1)
    either_order: tl.constexpr = (1, 0)
    block_ptr = tl.make_block_ptr(base=x_ptr, shape=absurd_shape, strides=(2, 3), offsets=(0, 0), block_shape=(2, 2), order=either_order)
    rng = tl.arange(0, 2)
    offsets_2d = rng[:, None] + rng[None, :] * 5  # [0, 1]^T + [0, 5] = [[0, 1], [5, 6]]
    tl.store(o_ptr + offsets_2d, tl.load(block_ptr))

x = torch.arange(10).cuda()
o = x * 0

compiled_kernel = block_kernel[(1,)](x, o)

# to get all the codegen keys
print(compiled_kernel.asm.keys())
# dict_keys(['llir', 'ttgir', 'ttir', 'ptx', 'cubin'])

# print ir beneath
print("===================== ttir =====================")
print(compiled_kernel.asm["ttir"])    # triton ir
print("===================== ttgir =====================")
print(compiled_kernel.asm["ttgir"])   # triton gpu ir
print("===================== llir =====================")
print(compiled_kernel.asm["llir"])    # llvm ir
print("===================== ptx =====================")
print(compiled_kernel.asm["ptx"])     # ptx ir
with open("./tmp/cubin_data.o", "wb") as f:
    f.write(compiled_kernel.asm["cubin"])
print("===================== cubin =====================")
print(get_sass(compiled_kernel.asm["cubin"]))


