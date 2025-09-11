import torch
import triton
import triton.language as tl 

"""
    这是一个基础的FlashAttention v2实现
    参考：
    https://github.com/hkproj/triton-flash-attention/blob/main/triton/flash_attention.py
"""

"""
    inner loop
"""
@tl.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # 首先先处理causal attention的情况
    if STAGE == 1:
        # causal attention：左下角block范围
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q 
    elif STAGE == 2:
        # causal attention：对角线block的处理
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
    else: 
        # 非causal attention的情况
        lo, hi = 0, SEQ_LEN 

    # 定位K和V
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # 遍历K和V
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # compiler 优化hint
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # 计算qk
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        # 更新rowmax，对于causal attention，需要使用mask
        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])  # 2维度
            # query的dimension不能超过key和value
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            # 部分row max
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, 1)

        alpha = tl.math.exp(m_i - m_ij)     # 调整因子
        l_i= l_i * alpha +l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        # 计算如下内容： O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # 移动K和V 的block
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i

"""
    前向pass的outer loop
    1. Q,K,V分别切为BLOCK_SIZE_Q和BLOCK_SIZE_KV
    2. Q是outer 循环，load Q block到SRAM
    3. K和V是inner循环，调用inner loop函数
"""
@tl.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,     # 需要autotune调优
    BLOCK_SIZE_KV: tl.constexpr,    # 需要autotune调优
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # 获取block index
    block_index_q = tl.program_id(0)                   # seq 层面的并行
    index_batch_head = tl.program_id(1)                # batch * num_head并行
    index_batch = index_batch_head // NUM_HEADS        # 定位具体的batch
    index_head = index_batch_head % NUM_HEADS          # 定位具体的head

    # qkv的四个维度，定位好前两个维度
    qkv_offset = (
        index_batch.to(tl.float16) * stride_Q_batch.to(tl.float16)
        + index_head.to(tl.float16) * stride_Q_head.to(tl.float16)
    )

    # 定位Q
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,                        # 前两个维度对应好
        shape=(SEQ_LEN, HEAD_DIM),                  # 一个block是后两个维度
        strides=(stride_Q_seq, stride_Q_dim), 
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),  # 两个维度，SEQ维度需要定位
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),       # 的模型的HEAD_DIM一般为64/128，较小
        order=(1,0),                                # 第1维度线程变化快，第0维度慢
    )

    # 定位K
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),                  # 注意shape反转
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    # 定位V
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    # 定位O
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # 计算Q和kv的偏移
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # 初始化一个block要计算的m_i, l_i以及output
    # m_i 是logsumexp的值
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i 是running sum
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # 累加
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # 加载Q block到SRAM，以重用
    Q_block = tl.load(Q_block_ptr)

    # stage3表示是causal attention
    if STAGE == 1 or STAGE == 3:
        # 这段代码对于causal attention完成左下角，对于non causal完成全部
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # 针对对角线的block，做相应的处理
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    # softmax(xi) = (exp^(xi-mi)) / (li) = exp^(xi-mi-log(li))
    # 所以反向传播过程中，只要我们保存mi + log(li)，即可做重计算 
    m_i += tl.math.log(l_i)       # 用于后续反向传播使用
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i) 
    tl.store(O_block_ptr, O_block.to(O.O.type.element_ty))

@tl.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,    # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    O_block = tl.load( # O [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        O 
        + index_batch_head * HEAD_DIM * SEQ_LEN
        +offs_q[:, None] * HEAD_DIM
        +offs_dim[None, :]
    )

    dO_block = tl.load( # O [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        O 
        + index_batch_head * HEAD_DIM * SEQ_LEN
        +offs_q[:, None] * HEAD_DIM
        +offs_dim[None, :]
    ).to(tl.float32)

    # 计算D block 
    D_block = tl.sum(dO_block * O_block, 1)      # (BLOCK_SIZE_Q,)
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)

@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV1, HEAD_DIM)
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # Shape: (BLOCK_KV1, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    # We access the Q as a transposed array, so that's why we treat offs_q as a column vector ans offs_dim as a row vector
    # This is equivalent to doing:
    # q_ptrs = Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration.
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # Iterates over the sequence dimension of the query
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load a block of Q
        qT_block = tl.load(qT_ptrs)
        # Load the logsumexp values for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)

        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = P^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        # We apply the softmax by using the logsumexp trick
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # Autoregressive masking.
            # mask is True for all values that DO NOT NEED TO BE MASKED
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )  # Shape: (BLOCK_KV1, BLOCK_Q1)
            # Replace all the masked values with 0.
            # In this case we do not need to mask with -Inf before applying the softmax since we already computed the normalization factors (stored in "m")
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        # According to the formula: dV_new = dV_old + P^T x dO, where x is the matrix multiplication
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # Delta = rowsum(O * dO) where * is the element-wise product
        Di = tl.load(D + offs_q)

        # dP = dO x V^T, so dP^T = V x dO^T
        # Where x is the matrix multiplication
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # We know that dS = P * (dP - Delta), so dS^T = P^T * (dP^T - Delta^T)

        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        # According to the formula on the paper: dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # Increment pointers.
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    # Write back dV.
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    # Write back dK.
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block) 

class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # 维度：BATCH_SIZE, NUM_HEAD, SEQ_LEN, HEAD_DIM
        # 维度检查
        HEAD_DIM_K, HEAD_DIM_Q = K.shape[-1], Q.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEAD, SEQ_LEN = Q.shape[0], Q.shape[1], Q.shape[2]
        
        assert HEAD_DIM_K == HEAD_DIM_Q and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)

        # 是否使用mask当query的index比key的index更大
        stage = 3 if causal else 1

        # 注意BATCH_SIZE和NUM_HEAD是block并行
        grid = lambda args: (
            tl.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEAD,
            1,
        )

        # M 是logsumexp，triton前向计算，用于反向传播
        M = torch.empty(
            (BATCH_SIZE, NUM_HEAD, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # triton内核代码实现
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,         # logsumexp 用于反向传播使用
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        process_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)

        # 计算Di
        _attn_bwd_preprocess[process_grid](
            O = O,
            dO = dO,
            D = D,
            SEQ_LEN = SEQ_LEN,
            BLOCK_SIZE_Q = BLOCK_SIZE_MACRO,
            HEAD_DIM = ctx.HEAD_DIM,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.causal else 1

        # 利用预处理计算出的Di来修正kv
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

# 单测试框架
def test_op(BATCH_SIZE, NUM_HEAD, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    # 创建输入张量
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEAD, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEAD, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEAD, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )

    softmax_scale= 1 / (HEAD_DIM ** 0.5)
    dO = torch.rand_like(Q)       
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).to(torch.float16)
    ref_O = torch.matmul(P, V)

    # 反向传播
    ref_O.backward(dO)
    ref_dV = V.grad.clone() # type: ignore
    ref_dK = K.grad.clone() # type: ignore
    ref_dQ = Q.grad.clone() # type: ignore

    # 注意V，K，Q的权重要清零
    V.grad = None
    K.grad = None   
    Q.grad = None

    # Triton实现

    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale)
    assert tri_out is not None
    tri_out = tri_out.to(torch.float16)

    tri_out.backward(dO)
    tri_dV = V.grad.clone() # type: ignore
    tri_dK = K.grad.clone() # type: ignore
    tri_dQ = Q.grad.clone() # type: ignore

    V.grad = None
    K.grad = None
    Q.grad = None

    # 检查梯度是否一致
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol), "V的梯度不一致"
    assert torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol), "K的梯度不一致"
    assert torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol), "Q的梯度不一致"
    # 检查输出是否一致
    assert torch.allclose(ref_O, tri_out, rtol=rtol, atol=atol), "输出不一致"

if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEAD=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEAD=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    print("Passed")
    

