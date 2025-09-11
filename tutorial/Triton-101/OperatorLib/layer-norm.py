import triton 
import triton.language as tl 
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()
# print(DEVICE)     result is cuda:0

# compare with apex lib: https://github.com/NVIDIA/apex
try:
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    print("apex not found, please install apex first")
    HAS_APEX = False

#   Layer Normalization Calculation:
#   1. Forward pass: 
#   y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#   2. Backward pass: 
#   \nabla_{x} = \frac{1}{\sigma}\Big( \nabla_{y} \odot w - \underbrace{ \big( \frac{1}{N} \hat{x} \cdot (\nabla_{y} \odot w) \big) }_{c_1} \odot \hat{x} - \underbrace{ \frac{1}{N} \nabla_{y} \cdot w }_{c_2} \Big)

# forward pass
@triton.jit
def _layer_norm_fwd_fused(
    X, Y, Weight, Bias, Mean, Rstd, eps,
    x_stride, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    X += pid * x_stride
    Y += pid * x_stride
    _mean = tl.zeros([BLOCK_SIZE,], dtype=tl.float32)
    mean = 0
    _var = tl.zeros([BLOCK_SIZE,], dtype=tl.float32)

    # calculate mean
    for i in range(0, N, BLOCK_SIZE):
        col = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + col, mask=col < N, other=0.).to(tl.float32)
        _mean += x 

    mean = tl.sum(_mean, axis=0) / N 
    tl.store(Mean + pid, mean)
    
    # calculate variance
    for i in range(0, N, BLOCK_SIZE):
        col = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + col, mask=col < N, other=0.).to(tl.float32)
        x = tl.where(col < N, x-mean, 0.)
        _var += x * x

    var = tl.sum(_var, axis=0) / N 
    rstd = 1 / tl.sqrt(var+eps)
    tl.store(Rstd + pid, rstd)

    # calculate output
    for i in range(0, N, BLOCK_SIZE):
        col = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + col, mask=col < N, other=0.).to(tl.float32)
        weight = tl.load(Weight + col, mask=col < N).to(tl.float32)
        bias = tl.load(Bias + col, mask=col < N).to(tl.float32)
        x = tl.where(col < N, (x-mean) * rstd * weight + bias, 0.)
        tl.store(Y + col, x, mask=col < N)

# backward pass for dx
@triton.jit
def _layer_norm_bwd_stage1(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0)

# backward pass for db and dw
@triton.jit
def _layer_norm_bwd_stage2(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # get Column ID
    pid = tl.program_id(axis=0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # compute BLOCK_SIZE_M * BLOCK_SIZE_N partial weights
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    # accumulate the partial weights
    for i in tl.range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = (rows[:, None] * N) + (cols[None, :])
        dw += tl.load(DW + offs, mask=mask, other=0).to(tl.float32)
        db += tl.load(DB + offs, mask=mask, other=0).to(tl.float32)

    # reduce partial weights
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)

    # Final DB is (1, BLOCK_SIZE_N) as w and b are the same over rows
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


# Unit tests

# Benchmark
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have Less than 64KB per feature.
# Specifically, one can set :code:`'mode': 'backward'` to benchmark the backward pass.

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # a block calculates the mean and variance for one row
        mean = torch.empty((M, ), dtype=torch.float32, device=DEVICE)
        rstd = torch.empty((M, ), dtype=torch.float32, device=DEVICE)

        # block_size should not exceed shared memory size
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _layer_norm_fwd_fused[(M, )](
            x_arg, y, weight, bias, mean, rstd, eps,
            x_arg.stride(0), N,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            num_ctas=1
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y
    
    @staticmethod
    def backward(ctx, dy):
        # get the saved tensors from forward pass
        x, w, b, m, v = ctx.saved_tensors
        # heuristic search for parallel reduction
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        # This three tensors are used to store temp
        # for stage1
        locks = torch.zeros(2*GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)

        # Stage2 tensors
        dx = torch.empty_like(dy)
        dw = torch.empty((N,), dtype=w.dtype, device=DEVICE)
        db = torch.empty((N,), dtype=w.dtype, device=DEVICE)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # inputs: [M, N]
        _layer_norm_bwd_stage1[(M,)](
            dx, dy, _dw, _db, x, w, m, v, locks, 
            x_arg.stride(0), N,
            BLOCK_SIZE_N = ctx.BLOCK_SIZE,
            GROUP_SIZE_M = GROUP_SIZE_M,
            num_warps = ctx.num_warps
        )

        # cautious: grid should be two dimensional
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        # inputs: [Group_size_m, N]
        _layer_norm_bwd_stage2[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,
            BLOCK_SIZE_M = 32,
            BLOCK_SIZE_N = 128, num_ctas = 1
        )

        return dx, None, dw, db, None

layer_norm = LayerNorm.apply

def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_layer_norm(1151, 8192, torch.float16)
bench_layer_norm.run(save_path='.', print_data=True)