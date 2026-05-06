import argparse
import time

import torch
import triton
import triton.language as tl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch two Triton kernels with block_size=512 and 256 for NCU comparison."
    )
    parser.add_argument(
        "--mode",
        choices=("both", "bs512", "bs256"),
        default="both",
        help="Which kernel configuration to launch.",
    )
    parser.add_argument(
        "--n-elements",
        type=int,
        default=512 * 1024 * 1024,
        help="Total element count. Must be divisible by 512 so both kernels run without masks.",
    )
    parser.add_argument(
        "--num-warps",
        type=int,
        default=2,
        help="num_warps passed to both kernels.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup launches per selected kernel before the measured loop.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Measured launches per selected kernel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch RNG seed for reproducible inputs.",
    )
    parser.add_argument(
        "--profile-api",
        action="store_true",
        help="Bracket measured launches with cudaProfilerStart/cudaProfilerStop for ncu --profile-from-start off.",
    )
    return parser.parse_args()


@triton.jit
def triton_kernel_add_bs512(
    a_ptr,
    b_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets).to(tl.float32)
    tl.store(out_ptr + offsets, a + b)


@triton.jit
def triton_kernel_add_bs256(
    a_ptr,
    b_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets).to(tl.float32)
    tl.store(out_ptr + offsets, a + b)


def cuda_profiler_start() -> None:
    torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    torch.cuda.cudart().cudaProfilerStop()


def launch_kernel(kernel, block_size: int, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, num_warps: int) -> None:
    grid = (a.numel() // block_size,)
    kernel[grid](a, b, out, BLOCK_SIZE=block_size, num_warps=num_warps)


def run_case(name: str, kernel, block_size: int, args: argparse.Namespace, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    for _ in range(args.warmup):
        launch_kernel(kernel, block_size, a, b, out, args.num_warps)
    torch.cuda.synchronize()

    if args.profile_api:
        cuda_profiler_start()

    start = time.perf_counter()
    for _ in range(args.iters):
        launch_kernel(kernel, block_size, a, b, out, args.num_warps)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3

    if args.profile_api:
        cuda_profiler_stop()

    bytes_per_iter = a.numel() * (a.element_size() + b.element_size() + out.element_size())
    gib_per_s = (bytes_per_iter * args.iters) / (elapsed_ms / 1e3) / (1024 ** 3)
    print(f"{name}: block_size={block_size}, iters={args.iters}, elapsed_ms={elapsed_ms:.3f}, approx_GiBps={gib_per_s:.2f}")


def main() -> None:
    args = parse_args()
    if args.n_elements % 512 != 0:
        raise ValueError("--n-elements must be divisible by 512.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    a = torch.randn(args.n_elements, device=device, dtype=torch.float32)
    b = torch.randn(args.n_elements, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(a)

    selected = []
    if args.mode in ("both", "bs512"):
        selected.append(("triton_kernel_add_bs512", triton_kernel_add_bs512, 512))
    if args.mode in ("both", "bs256"):
        selected.append(("triton_kernel_add_bs256", triton_kernel_add_bs256, 256))

    for name, kernel, block_size in selected:
        run_case(name, kernel, block_size, args, a, b, out)


if __name__ == "__main__":
    main()
