# Triton Debugger

## Print in Triton
`static_print` is for compile time print log, and `device_print` is for runtime print log.  
```python
tl.static_print(f"BLOCK_SIZE: {BLOCK_SIZE}") 
tl.device_print("pid: ", pid)
```

`static_assert` and `device_assert` also exists for assertion.

## Pdb debugger
For debugging python code in upper layer, we can use pdb. First `import pdb`, then use `pdb.set_trace()` to make a breakpoint in source code. 
> To be able to debug triton kernel, `TRITON_INTERPRET=1` we should turn to triton intepreter mode.

## C++ code debugger
To debug c++ code behind python wrapper, we can simply use `gdb` tool. Following command is a way to debug both python and c++ parts:

```shell
gdb --args python xxx.py
```

> to be able to step into c++ part, make sure triton is installed in debug mode. See build docs for how to install from source.

make a break point in MLIR source code:
```shell
b /home/douliyang/large/triton-workspace/triton-tutorial/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:784
```


## IR Dump 
Below are some useful commands that can help debug Triton Kernel.  

* `MLIR_ENABLE_DUMP=1` to help dump ir code before and after each mlir pass transformation.  
    ```shell
    MLIR_ENABLE_DUMP=1 pythonvector_add.py &> ir_dump.log
    ```
* `LLVM_IR_ENABLE_DUMP=1` for llvm pass dump.
* `TRITON_LLVM_DEBUG_ONLY` enables constrain dump towards certain pass. eg. `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"`
* `TRITON_PRINT_AUTOTUNING=1` dump metadata after auto tune.

refer to [Triton knobs.py](https://github.com/triton-lang/triton/blob/main/python/triton/knobs.py) for detailed dump commands.

## Pdb & GDB commands
Below is from chatgpt  

### 🐍 pdb (Python Debugger) Commands
| Command     | Description                                      |
|-------------|--------------------------------------------------|
| `l`         | List source code around the current line         |
| `n`         | Next line, without stepping into function calls  |
| `s`         | Step into a function call                        |
| `r`         | Continue until the current function returns      |
| `c`         | Continue execution until a breakpoint or exit    |
| `b`         | Set a breakpoint                                 |
| `cl`        | Clear breakpoints                                |
| `p expr`    | Print the value of an expression                 |
| `pp expr`   | Pretty-print the value of an expression          |
| `a`         | Display function arguments                       |
| `! stmt`    | Execute a Python statement                       |
| `q`         | Quit the debugger                                |
| `w` / `where` | Show the current call stack                   |
| `up`        | Move up the call stack                           |
| `down`      | Move down the call stack                         |
| `help`      | Show help message or help for a specific command |

### 🧰 gdb (GNU Debugger) Commands
| Command          | Description                                           |
|------------------|-------------------------------------------------------|
| `start`          | Start the program and stop at `main`                 |
| `run` / `r`      | Run the program with arguments                       |
| `b`              | Set a breakpoint                                     |
| `info b`         | List current breakpoints                             |
| `d` / `delete`   | Delete breakpoints                                   |
| `c` / `continue` | Continue execution                                   |
| `s` / `step`     | Step into the next instruction                       |
| `n` / `next`     | Step over function calls                             |
| `finish`         | Run until the current function returns               |
| `bt`             | Show backtrace (call stack)                          |
| `up`             | Move up one stack frame                              |
| `down`           | Move down one stack frame                            |
| `p expr`         | Print the value of an expression                     |
| `display expr`   | Auto-print an expression each time the program stops |
| `x addr`         | Examine memory at an address                         |
| `info locals`    | Show local variables                                 |
| `info args`      | Show function arguments                              |
| `set var`        | Set the value of a variable                          |
| `list` / `l`     | Show source code near the current line               |
| `watch expr`     | Stop when the value of an expression changes         |
| `set pagination off` | Disable output paging                          |
| `q` / `quit`     | Exit GDB                                             |

For MLIR developer, we may want to debug each MLIR pass. Following is a vscode `launch.json` configuration which helps debug rewrite_tensor_ptr pass in ttir level:  

```json
{
    "name": "triton-opt debug rewrite-tensor",
    "type": "cppdbg",
    "request": "launch",
    "program": "/home/douliyang/large/triton-workspace/triton-tutorial/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt",
    "args": [
        "--triton-rewrite-tensor-pointer",
        "/home/douliyang/large/triton-workspace/triton-tutorial/tutorial/Triton-101/DeepDive/ttir/test/simple_test.mlir",
        "-debug"
    ],
    "stopAtEntry": false,
    "cwd": "${workspaceFolder}",
    "externalConsole": false,
    "environment": [
    {
        "name": "LD_PRELOAD",
        "value": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    },
    {
        "name": "PATH",
        "value": "/home/douliyang/large/triton-workspace/triton-tutorial/build/cmake.linux-x86_64-cpython-3.11/bin:${env:PATH}"
    },
    {
        "name": "PYTHONPATH",
        "value": "/mnt/home/douliyang/triton-workspace/triton-tutorial/python:${env:PYTHONPATH}"
    },
    {
        "name": "TRITON_INTERPRET",
        "value": "1"
    }
    ],
    "MIMode": "gdb",
    "miDebuggerPath": "/usr/bin/gdb",
    "setupCommands": [
        {
            "description": "Enable pretty-printing for gdb",
            "text": "-enable-pretty-printing",
            "ignoreFailures": true
        }
    ]
},
```
