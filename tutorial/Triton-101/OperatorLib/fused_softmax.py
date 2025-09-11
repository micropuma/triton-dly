import torch 

import triton 
import triton.language as tl 
from triton.runtime import driver  

#===================================== 获取device信息 =====================================
DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')
#===================================== pytorch实现 =====================================
def torch_softmax(x):
    """
    使用PyTorch的softmax函数实现
    """
    return torch.nn.functional.softmax(x, dim=-1)

def native_softmax(x):
    """
    使用python手工实现
    """

    x_max= x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret

#===================================== 实现一个triton kernel =====================================
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start= tl.program_id(0)       # 当前线程块的ID
    row_step = tl.num_programs(0)     # 总线程块数量

    # 启动多阶段流水线
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # 计算每一行的起始地址
        # 并计算每一行的列偏移
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offset = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offset
        mask = col_offset < n_cols

        # 读取输入数据
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # 写入输出数据
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offset
        tl.store(output_ptrs, softmax_output, mask=mask)
        
#===================================== softmax 对外接口 =====================================
# 获取设备属性
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

# 对外softmax接口
def softmax(x):
    n_rows, n_cols = x.shape

    # BLOCK_SIZE以此计算整个行，同时要是2的幂次方
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 根据smem大小来确定num_stages
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # 一个block多少个warp，可以人为设定来达到优化目的
    num_warps = 8

    # 开辟输出
    y = torch.empty_like(x)

    # 通过warmup来获取kernel的相关信息
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    # 计算occupanc
    # Occupancy 表示单个 SM 上可同时执行的 Warp 数量，受 ​​寄存器数量​​ 和 ​​共享内存容量​​ 
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        # 寄存器限制：总寄存器数 / (每个线程寄存器 × Warp 大小 × 每个实例 Warp 数)
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    # block数量不能超过行数
    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y.data_ptr(), x.data_ptr(), x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y

#===================================== 测试代码 =====================================
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch_softmax(x)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


