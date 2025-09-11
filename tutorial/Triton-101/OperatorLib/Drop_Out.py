'''
This file shows how to implement a memory efficient dropout layer in Triton.
'''

import torch
import tabulate

import triton 
import triton.language as tl 

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ================================== naive implementation =====================================
# Triton kernel for dropout
@triton.jit
def _dropout(x, mask_ptr, output_ptr, p, nelem, BLOCK_SIZE: tl.constexpr):
    # compute the index of the current element
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = block_start < nelem

    # load the input tensor and mask
    x = tl.load(x + offsets, mask=mask, other=0.0)
    x_mask = tl.load(mask_ptr + offsets, mask=mask, other=0)
    # apply dropout
    output = tl.where(x_mask, x / (1 - p), 0.0)
    # store the output tensor
    tl.store(output_ptr + offsets, output, mask=mask)

# wrapper for the dropout kernel
def dropout(x, mask, p=0.5):
    output = torch.empty_like(x, device=DEVICE)
    assert x.is_contiguous(), "Input tensor must be contiguous"
    nelem = x.numel()

    # meta infer BLOCK_SIZE is auto configured by Triton
    grid = lambda meta: (triton.cdiv(nelem, meta['BLOCK_SIZE']),)
    _dropout[grid](x, mask, output, p, nelem, BLOCK_SIZE=1024)

    return output


# ========================================= testbench =========================================
X = torch.randn(size=(10,), device=DEVICE)
p = 0.5
# use rand instead of randn
mask = (torch.rand(size=(10,), device=DEVICE) > p).to(torch.int32)
output = dropout(X, mask, p)

print(tabulate.tabulate([
    ['Input'] + X.tolist(),
    ['Mask'] + mask.tolist(),
    ['Output'] + output.tolist(),
]))

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


x = torch.randn(size=(10, ), device=DEVICE)
# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))

