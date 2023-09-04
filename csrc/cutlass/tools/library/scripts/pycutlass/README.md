# PyCUTLASS: CUTLASS Python Interface

PyCUTLASS is a python interface of CUTLASS C++ template library. PyCUTLASS takes user-defined operation descriptions, emits C++ code, and compiles it with `nvcc` or `nvrtc`. It also provides wrappers for user-provide arguments from [numpy](https://numpy.org/), [torch](https://pytorch.org/), and [cupy](https://github.com/cupy/cupy) and encode them to kernel's parameters.

```python
import pycutlass
from pycutlass import *
import torch

pycutlass.get_memory_pool(2**8, 2**32)

math_inst = MathInstruction(
    [1, 1, 1], cutlass.float32, cutlass.float32, cutlass.float32,
    cutlass.OpClass.Simt, MathOperation.multiply_add
)

tile_description = TileDescription(
    [128, 128, 8], 4, [2, 4, 1],
    math_inst
)

A = TensorDescription(
    cutlass.float32, cutlass.RowMajor, 1
)

B = TensorDescription(
    cutlass.float32, cutlass.RowMajor, 1
)

C = TensorDescription(
    cutlass.float32, cutlass.RowMajor, 1
)

epilogue_functor = LinearCombination(cutlass.float32, 1, cutlass.float32, cutlass.float32)

operation = GemmOperationUniversal(
    arch=80, tile_description=tile_description,
    A=A, B=B, C=C, 
    epilogue_functor=epilogue_functor, 
    swizzling_functor=cutlass.IdentitySwizzle1
)

pycutlass.compiler.add_module([operation,])

problem_size = cutlass.gemm.GemmCoord(512, 256, 128)

tensor_A = torch.ceil(torch.empty(size=(problem_size.m(), problem_size.k()), dtype=torch.float32, device="cuda").uniform_(-8.5, 7.5))
tensor_B = torch.ceil(torch.empty(size=(problem_size.k(), problem_size.n()), dtype=torch.float32, device="cuda").uniform_(-8.5, 7.5))
tensor_C = torch.ceil(torch.empty(size=(problem_size.m(), problem_size.n()), dtype=torch.float32, device="cuda").uniform_(-8.5, 7.5))
tensor_D = torch.empty_like(tensor_C)


alpha = 1.0
beta = 0.0

arguments = GemmArguments(
    operation=operation, problem_size=problem_size,
    A=tensor_A, B=tensor_B, C=tensor_C, D=tensor_D,
    output_op=operation.epilogue_type(alpha, beta),
    gemm_mode=cutlass.gemm.Mode.Gemm, split_k_splices=1
)

operation.run(arguments)

arguments.sync()

tensor_D_ref = alpha * tensor_A @ tensor_B + beta * tensor_C

assert torch.equal(tensor_D, tensor_D_ref)
```
PyCUTLASS also provides infrastructures for profiling, compiled artifact management, and pool memory manager 

## Supported Features
PyCUTLASS currently supports following operations:
* GEMM with mode {Serial, Parallel Split K, Batched GEMM, Array GEMM}, op class {SIMT, TensorCore}, data type {int8, f16, bf16, f32, f64}, layout {RowMajor, ColumnMajor, Row/ColumnMajorInterleaved<32> for int8}, math operation {MultiplyAdd, MultiplyAddFastF16, MultiplyAddFastBF16, MultiplyAddFastF32}, swizzling functions {IdentitySwizzle<1,2,4,8>, HorizontalSwizzle, BatchedIdentitySwizzle}, and epilogue {LinearCombination, LinearCombinationClamp}
* GEMM grouped with op class {SIMT, TensorCore}, data type {int8, f16, bf16, f32, f64}, layout {RowMajor, ColumnMajor}, math operation {MultiplyAdd, MultiplyAddFastF16, MultiplyAddFastBF16, MultiplyAddFastF32}, scheduling mode {Host, Device}, and epilogue {LinearCombination, LinearCombinationClamp}.
* Conv2d with {Fprop, Dgrad, Wgrad}, op class {SIMT, TensorCore}, data type {int8, f16, bf16, f32, f64}, layout {Tensor NHWC, TensorNC32HW32 and TensorC32RSK for int8}, math operation {MultiplyAdd, MultiplyAddFastF16, MultiplyAddFastBF16, MultiplyAddFastF32}, split-k mode {Parallel, Serial}, and epilogue {LinearCombination, LinearCombinationClamp}

The tiling size of above operations can also be customized.

## Installation

### Using Docker
We recommend using one of our provided Docker images for using PyCUTLASS.

**To run CUTLASS 3 GEMM kernels targetting the NVIDIA Hopper architecture via PyCUTLASS,** you can use an included [Dockerfile](docker/Dockerfile-cuda12.0) based on the NGC CUDA 12.0 container:
```shell
docker build -t pycutlass-cuda12.0:latest -f docker/Dockerfile-cuda12.0 .
docker run --gpus all -it --rm pycutlass-cuda12.0:latest
```
Note that this Docker container does not include CuPy or PyTorch, and, thus, will not be able to run PyCUTLASS examples that
leverage these packages.

**To run CUTLASS 2.x kernels targetting pre-SM90 architectures via PyCUTLASS,** you can use an included [Dockerfile](docker/Dockerfile-cuda11.8-pytorch) based on an NGC PyTorch container:
```shell
docker build -t pycutlass-cuda11.8-pytorch:latest -f docker/Dockerfile-cuda11.8-pytorch .
docker run --gpus all -it --rm pycutlass-cuda11.8-pytorch:latest
```

### Environment variables
PyCUTLASS requires two environment variables:
* `CUTLASS_PATH`: the root directory of CUTLASS. You can set this from the location at which you cloned CUTLASS via: `export CUTLASS_PATH=$(pwd)`.
* `CUDA_INSTALL_PATH`: the directory where cuda toolkit is installed. If running in bash with `nvcc` installed under a CUDA toolkit, you can set this to the location of your `nvcc` installation via: `export CUDA_INSTALL_PATH=$(which nvcc | awk -F'/bin/nvcc' '{print $1}')`

After setting these two environment variables, PyCUTLASS can be installed with 
```shell
cd $CUTLASS_PATH/tools/library/scripts/pycutlass && bash build.sh
```

## Examples
Examples can be found in [$CUTLASS_PATH/examples/40_cutlass_py](examples/40_cutlass_py)

## Test
The test cases are listed in `$CUTLASS_PATH//tools/library/scripts/pycutlass/test`. The unit test can be run with
```shell
# Each of these tests are only supported on devices with compute capability of SM80. For other devices,
# see the basic examples in $CUTLASS_PATH/examples/40_cutlass_py
cd $CUTLASS_PATH/tools/library/scripts/pycutlass/test/unit && python test_sm80.py
cd $CUTLASS_PATH/tools/library/scripts/pycutlass/test/example && bash run_all_example.sh
```

## build documentation
Run
```shell
bash build_doc.sh
```


## Troubleshooting

### Issue 1: permission denied
Building PyCUTLASS requires installing dependencies to python. So conda could an option if you don't have permission.

### Issue 2: rmm: module not found
PyCUTLASS manages the device memory with [RMM](https://github.com/rapidsai/rmm). Our `build.sh` automatically pull the [rmm branch-22.08](https://github.com/rapidsai/rmm/tree/branch-22.08) from github and build it from source. The rmm is allocated at `$CUTLASS_PATH/tools/library/scripts/pycutlass/rmm`. It requires `cmake > 3.20.1`. If the build fails, it can be manually fixed with the following steps:
```shell
cd $CUTLASS_PATH/tools/library/scripts/pycutlass/rmm && ./build.sh librmm rmm

cd $CUTLASS_PATH/tools/library/scripts/pycutlass/rmm/python
python setup.py build_ext --inplace
python setup.py install
```
To test whether rmm is successfully installed, try `import rmm`. For other issues related to rmm, please check https://github.com/rapidsai/rmm/issues. 
