import torch
import triton
import triton.language as tl
import os
from IPython.core.debugger import set_trace

from helpers import check_tensors_gpu_ready, cdiv, print_if   # import debugging tools

os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

"""
    reference: https://github.com/gpu-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb
"""

# =============================== Triton Kernel =======================================
# # This is the triton kernel:

# The triton.jit decorator takes a python function and turns it into a triton kernel, which is run on the GPU.
# Inside this function only a subset of all python ops are allowed.
# E.g., when NOT simulating, we can't print or use breakpoints, as these don't exist on the GPU. 
@triton.jit
# When we pass torch tensors, they are automatically converted into a pointer to their first value
# E.g., above we passed x, but here we receive x_ptr
def copy_k(x_ptr, z_ptr, n, bs: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * bs + tl.arange(0, bs)  # compute the offsets from the pid 
    mask = offs < n
    x = tl.load(x_ptr + offs, mask)  # load a vector of values, think of `x_ptr + offs` as `x_ptr[offs]`
    tl.store(z_ptr + offs, x, mask)  # store a vector of values

    print_if(f'pid = {pid} | offs = {offs}, mask = {mask}, x = {x}', '')

# # This is a normal python function, which launches the triton kernels
def copy(x, bs, kernel_fn):
    z = torch.zeros_like(x)
    check_tensors_gpu_ready(x, z)
    n = x.numel()
    n_blocks = cdiv(n, bs)
    grid = (n_blocks,)  # how many blocks do we have? can be 1d/2d/3d-tuple or function returning 1d/2d/3d-tuple

    # launch grid!
    # - kernel_fn is the triton kernel, which we write below
    # - grid is the grid we constructed above
    # - x,z,n,bs are paramters that are passed into each kernel function
    kernel_fn[grid](x,z,n,bs)

    return z   

# =============================== Main Function =======================================
x = torch.tensor([1,2,3,4,5,6], device='cuda')
z = copy(x, bs=2, kernel_fn=copy_k)
print(z)