'''
Matrix multiplication kernel.

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.

如下是一个基本的矩阵乘的伪代码实现思路
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
  # Do in parallel
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
'''

import torch

import triton
import triton.language as tl

# ================================= 设备配置 =================================

# 获取当前设备
DEVICE = triton.runtime.driver.active.get_active_torch_device()

# 判断当前GPU是支持cuda还是支持rocm
def is_cuda():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'cuda'

def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

# triton支持自动调优机制
# Group Size是为了L2 cache优化机制准备的
def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

# Group size 扩展成二维的版本
def get_cuda_autotune_config_2D():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 32, 'GROUP_SIZE_N': 32}, num_stages=4, num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]

def get_auto_config():
    if is_cuda():
        return get_cuda_autotune_config()
    elif is_hip_cdna2():
        return get_hip_autotune_config()
    else:
        raise RuntimeError('Unsupported backend for auto-tuning')
    
'''
这个教程，我们尝试实现的是矩阵乘法+bias加法的融合算子操作
'''
# ================================= 核心内核 =================================

# leaky_relu算子的基础实现
def leaky_relu(x):
    return tl.where(x >= 0, x, x * 0.01)

# 矩阵乘法算子
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs = get_auto_config(),
    key = ['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # 计算pid映射到的block范围
    pid = tl.program_id(axis=0)
    # 计算x轴和y轴各自有多少个blokc
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    # 计算一个group多少个blocks
    # 一个group是group_size行
    num_groups_blocks = GROUP_SIZE_M * num_blocks_n
    # 计算当前block在group中的位置
    group_id = pid // num_groups_blocks
    # 定位当前block的m坐标
    first_pid_m = group_id * GROUP_SIZE_M
    # 计算当前group在m维度的长度，可能不是满维度，即不是GROUP_SIZE_M    
    group_size_m = min(GROUP_SIZE_M, num_blocks_m - first_pid_m)
    # 计算当前block在group中的坐标
    # 这里有个trick，是column order
    pid_m = first_pid_m + ((pid % num_groups_blocks) % group_size_m)
    pid_n = (pid % num_groups_blocks) // group_size_m

    # 计算指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # 累加器是fp32位置，由于fp16相乘需要fp32来承接
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载数据
        a = tl.load(a_ptrs, mask = offs_k[None, :] < K - k*BLOCK_SIZE_K, other = 0.0)
        b = tl.load(b_ptrs, mask = offs_k[:, None] < K - k*BLOCK_SIZE_K, other = 0.0)
        acc = tl.dot(a,b,acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 如果有激活函数，则应用激活函数
    if ACTIVATION == "leaky_relu":
        acc = leaky_relu(acc)
    c = acc.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)  

# 矩阵乘法算子 2D L2 caching优化版本
@triton.autotune(
    configs = get_cuda_autotune_config_2D(),
    key = ['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_advanced(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # 计算pid映射到的block范围
    pid = tl.program_id(axis=0)
    # 计算x轴和y轴各自有多少个blokc
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    # 计算一个group多少个blocks
    num_groups_blocks = GROUP_SIZE_M * GROUP_SIZE_N
    # 计算当前block在group中的位置，这个group_id是二维的
    group_id = pid // num_groups_blocks
    group_id_m = group_id // GROUP_SIZE_N
    group_id_n = group_id % GROUP_SIZE_N

    # 定位当前block的m坐标和n坐标
    first_pid_m = group_id_m * GROUP_SIZE_M
    first_pid_n = group_id_n * GROUP_SIZE_N

    # 计算当前group在m维度的长度，可能不是满维度，即不是GROUP_SIZE_M    
    # 在n维度同理
    group_size_m = min(GROUP_SIZE_M, num_blocks_m - first_pid_m)
    group_size_n = min(GROUP_SIZE_N, num_blocks_n - first_pid_n)

    # 计算当前block在group中的坐标
    # 这里有个trick，是column order
    pid_m = first_pid_m + ((pid % num_groups_blocks) % group_size_m)
    pid_n = first_pid_n + ((pid % num_groups_blocks) // group_size_m)

    # 计算指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # 累加器是fp32位置，由于fp16相乘需要fp32来承接
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载数据
        a = tl.load(a_ptrs, mask = offs_k[None, :] < K - k*BLOCK_SIZE_K, other = 0.0)
        b = tl.load(b_ptrs, mask = offs_k[:, None] < K - k*BLOCK_SIZE_K, other = 0.0)
        acc = tl.dot(a,b,acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 如果有激活函数，则应用激活函数
    if ACTIVATION == "leaky_relu":
        acc = leaky_relu(acc)
    c = acc.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask) 

# 整个融合算子的wrapper实现
def matmul(a, b, activation=""):
    # 矩阵维度检查
    assert a.shape[1] == b.shape[0], "Matrix dimensions do not match for multiplication"
    assert a.dtype == b.dtype, "Matrix data types do not match"
    assert a.is_contiguous(), "Matrix A must be contiguous in memory"

    M, K = a.shape
    K, N = b.shape

    # 开辟输出矩阵
    c = torch.empty((M, N), device = a.device, dtype=torch.float16)
    # 1D 发射kernel，每个block有对应的program
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    # 调用triton的kernel
    matmul_kernel[grid] (
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )

    return c

# ================================= 高级内核，2D L2 cache优化 =================================

# 先前的L2 cache只有M维度的group size，这次尝试两个维度的group size操作
def matmul_advanced(a, b, activation=""):
    # 矩阵维度检查
    assert a.shape[1] == b.shape[0], "Matrix dimensions do not match for multiplication"
    assert a.dtype == b.dtype, "Matrix data types do not match"
    assert a.is_contiguous(), "Matrix A must be contiguous in memory"

    M, K = a.shape
    K, N = b.shape

    # 开辟输出矩阵
    c = torch.empty((M, N), device = a.device, dtype=torch.float16)
    # 1D 发射kernel，每个block有对应的program
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    # 调用triton的kernel
    matmul_kernel_advanced[grid] (
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )

    return c

# ================================= 测试框架 =================================
# --------------- 正确性验证 ---------------
# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

torch.manual_seed(0)
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
triton_output = matmul(a, b)
triton_output_advanced = matmul_advanced(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"triton_output_advanced_with_fp16_inputs={triton_output_advanced}")
print(f"torch_output_with_fp16_inputs={torch_output}")
# Bigger tolerance for AMD CDNA2 devices.
# CDNA2 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 1e-2 if is_hip_cdna2() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
if torch.allclose(triton_output_advanced, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton advanced and Torch match")
else:
    print("❌ Triton advanced and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

# --------------- 性能测试 ---------------
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton", "triton_advanced"] if fp8_inputs else [ref_lib.lower(), "triton", "triton_advanced"],  # Add matmul_advanced to line_vals
            line_names=["Triton", "Triton Advanced"] if fp8_inputs else [ref_lib, "Triton", "Triton Advanced"],  # Add matmul_advanced to line_names
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],  # Style for matmul_advanced (red)
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))



@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]

    # Benchmark the reference library (e.g., cuBLAS or rocBLAS)
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)

    # Benchmark Triton matmul (original)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

    # Benchmark Triton matmul with 2D L2 cache optimization (advanced version)
    if provider == 'triton_advanced':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_advanced(a, b), quantiles=quantiles)

    # Calculate performance (in TFLOPS)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path='./matmul/')





