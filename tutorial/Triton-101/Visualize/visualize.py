import torch
import triton
import triton.language as tl
import tritonparse.structured_logging
import tritonparse.utils

# Initialize logging
log_path = "./logs/"
tritonparse.structured_logging.init(log_path, enable_trace_launch=True)

@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def tensor_add(a, b):
    n_elements = a.numel()
    c = torch.empty_like(a)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
    return c

# Example usage
if __name__ == "__main__":
    # Create test tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    
    # Execute kernel (this will be traced)
    c = tensor_add(a, b)
    
    # Parse the generated logs
    tritonparse.utils.unified_parse(
        source=log_path, 
        out="./parsed_output", 
        overwrite=True
    )
