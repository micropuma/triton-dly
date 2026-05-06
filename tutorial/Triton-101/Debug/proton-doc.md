# Proton 中文说明

## 简介

Proton 是 Triton 自带的轻量级性能分析器。它会记录程序上下文、用户标注的作用域、GPU Kernel 的运行时间，以及一些可选的性能指标，同时尽量控制运行时开销和 profile 文件大小。

## 安装

安装最新版本 Proton 的常用方式如下：

```bash
git clone https://github.com/triton-lang/triton
cd triton/python
pip install .
```

如果你不想编译 Proton，可以关闭构建开关：

```bash
TRITON_BUILD_PROTON=OFF pip install .
```

## 基本用法

### 分析整个函数

```python
import triton.profiler as proton

session_id = proton.profile(func, name="profile_name", context="python")(args)
```

`name` 是输出 profile 的路径前缀，`context` 控制 Kernel 的上下文标注方式。目前常用的是：

- `python`：显示 Python 文件、函数和行号
- `shadow`：显示用户通过 `proton.scope(...)` 标注的逻辑区域

### 分析一段代码区域

```python
import triton.profiler as proton

session_id = proton.start(name="profile_name", context="python")

# 暂停采集
proton.deactivate(session_id)

# 恢复采集
proton.activate(session_id)

# 写出 profile 并结束
proton.finalize()
```

### 使用 `scope` 标注逻辑区域

```python
import triton.profiler as proton

session_id = proton.start(name="profile_name", context="shadow")

with proton.scope("test0"):
    with proton.scope("test1"):
        foo[1,](x, y)

with proton.scope("test2"):
    foo[1,](x, y)

proton.finalize()
```

`scope` 也可以附带自定义指标，便于从更高层次理解程序性能：

```python
with proton.scope("test0", {"bytes": 1000}):
    with proton.scope("test1", {"bytes": 2000}):
        foo[1,](x, y)
```

Proton 会按 scope 聚合这些指标并写入 profile。

## Backend 与模式

Proton 支持三个后端：

- `cupti`：NVIDIA GPU，支持默认模式和 `pcsampling`
- `roctracer`：AMD GPU，只支持默认模式
- `instrumentation`：NVIDIA 和 AMD 都可用，适合更细粒度的自定义采样和内核内分析

默认情况下，Proton 会根据当前驱动自动选择 `cupti` 或 `roctracer`。

### 指令采样

在 NVIDIA GPU 上可以启用指令采样：

```python
import triton.profiler as proton

proton.start(name="profile_name", context="shadow", backend="cupti", mode="pcsampling")
```

这种模式通常会带来较高的端到端开销，但单个 GPU Kernel 的扰动相对较小。

### Instrumentation

`instrumentation` 后端适合分析 Kernel 内部行为。更详细的示例可以参考 `tutorials/intra_kernel/README.md`。

## 命令行用法

Proton 也可以直接分析 Python 脚本或 Pytest：

```bash
proton [options] script.py [script_args] [script_options]
proton [options] pytest [pytest_args] [script_options]
python -m triton.profiler.proton [options] script.py [script_args] [script_options]
proton --instrument=[instrumentation pass] script.py
```

命令行模式下，`proton.start` 和 `proton.finalize` 会在脚本执行前后自动调用。脚本里手写的 `proton.start/finalize` 会被忽略。命令行模式只支持一个 session，因此 `session_id` 只能是 `0`。

## 查看 Profile

默认输出格式是 Hatchet 可读的 `.hatchet` 文件，可以直接在终端中查看：

```bash
pip install llnl-hatchet
proton-viewer -m time/s <profile.hatchet>
```

注意：这里应安装 `llnl-hatchet`，不是 `hatchet`。

如果你想导出完整 trace，而不是聚合后的树，可以在启动时设置：

```python
import triton.profiler as proton

proton.start(name="profile_name", data="trace")
```

这会生成 Chrome Trace，可用 `chrome://tracing` 或 Perfetto 打开。

### 常用查看命令

```bash
proton-viewer -m time/ns,time/% <profile.hatchet> --print-sorted
proton-viewer -m tflop/s,time/s <profile.hatchet>
proton-viewer -m time/ns,cpu_time/ns <profile.hatchet>
```

更多参数可以直接运行：

```bash
proton-viewer -h
```

## 如何阅读 `proton-viewer` 输出

`proton-viewer` 打印出来的是一棵调用树。你传给 `-m` 的指标会按列显示。

例如：

```bash
proton-viewer -m tflop/s,time/s ./matmul.hatchet
```

表示：

- 第一列：`tflop/s`
- 第二列：`time/s`

### 先看树的层级

通常可以按下面的方式理解：

- `ROOT`：整个 profile 会话
- `matmul_1024_1024_1024`：一次 benchmark case
- `cublas` / `triton`：两种 provider
- `<autotune>`：Triton 首次执行时测试候选配置的阶段
- `matmul_<grid:...>_<warps:...>...`：实际被运行的某个 Triton Kernel 配置

在 `tutorials/matmul.py` 里，`<autotune>` 是显式包出来的 scope，用于记录调优阶段；最终正式执行的 Kernel 会出现在 `<autotune>` 外面。

### `time` 是 inclusive 时间

`time` 表示加速器时间，属于 inclusive 指标，也就是父节点时间包含子节点时间。

因此：

- `triton` 的总时间会包含 `<autotune>` 的时间
- `ROOT` 的时间会包含下面所有子树的时间

而 `cpu_time` 是 exclusive 的，不能和 `time` 直接按同样方式理解。

### `tflop/s` 是推导出来的，不是原始采样值

`tflop/s` 的计算方式本质上是：

```text
TFLOPS = flops / time
```

如果某个节点没有记录 `flops`，那么这个指标就算不出来，显示为 `nan`。

### `nan` 通常不是错误

在 Proton 里，`nan` 往往只是表示“这个节点没有对应指标”，常见情况包括：

- `ROOT` 这类组织性节点没有直接的 `flops`
- `triton`、`matmul_1024_1024_1024` 这种父 scope 只是容器
- 一些辅助 Kernel 没有记录 `flops`

所以：

- 看吞吐率时，重点看真正的 GEMM Kernel 或 `cublas` 节点
- 不要拿 `ROOT` 或纯容器 scope 的 `nan` 去做性能判断

### 为什么“时间看起来一样”，但 TFLOPS 不一样

例如你看到：

```text
43.214 0.008 cublas
41.867 0.008 matmul_<grid:144x1x1>_<cluster:1x1x1>_<warps:4>_<shared:30720>_<stages:4>
```

常见原因有两个：

1. 这两个 `0.008` 是显示时四舍五入后的结果，不代表底层原始时间完全相同。比如一个可能是 `0.0077` 秒，另一个可能是 `0.0081` 秒。
2. 第二行如果位于 `<autotune>` 下面，它只是一个候选 Triton 配置，不一定是最终被选中的正式执行 Kernel。

因此：

- 表面上都显示 `0.008 s`，真实值仍可能差几个百分点
- 这足以让 `tflop/s` 出现明显差异
- 比 Triton 性能时，应优先看 `<autotune>` 外面的最终 Kernel，而不是调优过程中的候选项

### 如何快速定位热点

如果你想先看谁最耗时：

```bash
proton-viewer -m time/ms,time/% ./matmul.hatchet --print-sorted
```

如果你想过滤掉很小的节点，只看主要 Kernel：

```bash
proton-viewer -m tflop/s,time/ms ./matmul.hatchet -t 0.001
```

如果你只想看包含某些名字的路径：

```bash
proton-viewer -i "matmul|cublas|triton" -m time/ms ./matmul.hatchet
```

## 指标命名约定

自定义指标建议使用：

```text
metric_name (unit) (type)
```

其中：

- `unit` 可选
- `type` 可选
- 默认类型是 inclusive

Proton 中常见的指标类型有：

- inclusive：默认类型，会沿父子作用域聚合
- exclusive：只属于当前作用域
- property：属性型指标，不做普通累加

## 常见建议

- 看端到端首次调用开销时，重点看 `triton` 父节点时间，因为它包含 autotune 和其他辅助开销。
- 看稳态 Kernel 性能时，重点看最终选中的 Triton Kernel 与 `cublas` 的 `tflop/s`。
- 小矩阵场景下，autotune 和辅助 Kernel 的一次性成本可能远大于 matmul 本体。
- 如果想看完整时间线，而不是聚合树，使用 `data="trace"` 导出 trace。
