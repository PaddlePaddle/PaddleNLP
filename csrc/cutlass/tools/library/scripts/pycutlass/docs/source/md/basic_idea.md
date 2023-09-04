# Basics of PyCUTLASS

PyCUTLASS handles the following things when launch the CUTLASS kernels
* Memory management
* Operation Description
* Code emission and compilation
* Arguments preprocessing
* Kernel launching
* Result Synchronization

## Memory management

PyCUTLASS uses [RMM](https://github.com/rapidsai/rmm) to manage device memory. At the begining of the program, call
```python
pycutlass.get_memory_pool({init_pool_size_in_bytes}, {max_pool_size_in_bytes})
```
We also provide functions to query the allocated size.
```python
bytes = get_allocated_size()
```


## Operation Description
PyCUTLASS provides operation description for GEMM, GEMM Grouped and Conv2d operations. These operation descriptions are assembled from four foundamental concepts
* Math Instruction: math instruction executed in GPU cores
* Tile Description: tiling sizes and pipeline stages
* Operand Description: data type, layout, memory alignment
* Epilogue Functor: epilogue function

### Math Instruction

The math instruction is defined as follows:
```python
math_inst = MathInstruction(
    {instruction_shape}, {element_a}, {element_b},
    {element_acc}, {opclass}, {math_operation}
)
```
The `{instruction_shape}` and `{opclass}` defines the instruction size and type. The table below lists valid combinations. `{element_a}`, `{element_b}` define the source operand data type for each instructions, and `{element_acc}` defines the accumulator type. The `{math_operation}` defines the math operation applied. 

|Opclass                   | element_a/element_b | element_acc     | instruction_shape | math_operation            |
| --                       | --                  | --              | --                | --                        |
| cutlass.OpClass.TensorOp | cutlass.float64     | cutlass.float64 | [8, 8, 4]         | MathOperation.multiply_add|
|                          | cutass.float32 cutlass.tfloat32, cutlass.float16 cutlass.bfloat16 | cutlass.float32 | [16, 8, 8] | MathOperation.multiply_add MathOperation.multiply_add_fast_f32 MathOperation.multiply_add_fast_f16 MathOperation.multiply_add_fast_bf16 |
|        | cutlass.float16 | cutlass.float16/cutlass.float32|[16, 8, 16]| MathOperation.multiply_add |
|        | cutlass.bfloat_16 | cutlass.float32 | [16, 8, 16]|MathOperation.multiply_add |
|        | cutlass.int8 | cutlass.int32 | [16, 8, 32] | MathOperation.multiply_add_saturate|
|cutlass.OpClass.Simt| cutlass.float64 | cutlass.float64 | [1, 1, 1] | MathOperation.multiply_add |
| | cutlass.float32 | cutlass.float32 | [1, 1, 1] | MathOperation.multiply_add |

The `cutlass.OpClass.TensorOp` indicates that the tensor core is used, while `cutlass.OpClass.Simt` uses the SIMT Core.

The `multiply_add_fast_f32` emulates fast accurate SGEMM kernel which is accelerated
using Ampere Tensor Cores. More details can be found in [examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm](examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm).

### Tile Description
The tile description describes the threadblock and warp tiling sizes, as well as the pipeline stages.
```python
tile_description = TileDescription(
    {threadblock_shape}, {stages}, {warp_count},
    math_inst
)
```
The `{threadblock_shape}` is a list of 3 integers `[Tile_M, Tile_N, Tile_K]` that defines the threadblock tiling size. `{stages}` defines the number of software pipeline stages ([detail](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)). `{warp_count}` defines the number of warps along `M`, `N`, and `K` dimension. I.e., with `{threadblock_shape}=[Tile_M, Tile_N, Tile_K]` and `{warp_count}=[W_M, W_N, W_K]`, the warp tile size would be `[Tile_M / W_M, Tile_N / W_N, Tile_K / W_K]`.

### Operand Description
The Operand Description defines the data type, layout, and memory alignment of input tensor A, B, and C. The output D shares the same attributes with C. The description is as follows:
```python
A = TensorDescription(
    {element_a}, {layout_a}, {alignment_a}
)

B = TensorDescription(
    {element_b}, {layout_b}, {alignment_b}
)

C = TensorDescription(
    {element_c}, {layout_c}, {alignment_c}
)
```
The table below lists the supported layout and data types for each operation
| Operation | data type | layout |
| --        | --        | --     |
| GEMM, GEMM Grouped     | cutlass.float64, cutlass.float32, cutlass.float16, cutlass.bfloat16 | cutlass.RowMajor, cutlass.ColumnMajor |
|           | cutlass.int8 | cutlass.RowMajor, cutlass.ColumnMajor, cutlass.RowMajorInterleaved32, cutlass.ColumnMajorInterleaved32|
| Conv2d Fprop, Dgrad, Wgrad | cutlass.float64, cutlass.float32, cutlass.float16, cutlass.bfloat16 | cutlass.TensorNHWC |
| Conv2d Fprop | cutlass.int8 | cutlass.TensorNHWC, cutlass.TensorNC32HW32, cutlass.TensorC32RSK32|

### Epilogue Functor
The epilogue functor defines the epilogue executed after mainloop.
We expose the following epilogue functors.
| Epilogue Functor | Remark |
| --               | --     |
| LinearCombination | $D=\alpha \times Accm + \beta \times C$ |
| LinearCombinationClamp | $D=\alpha \times Accm + \beta \times C$, Output is clamped to the maximum value of the data type output |
| FastLinearCombinationClamp | $D=\alpha \times Accm + \beta \times C$, only used for problem size $K\le 256$ for cutlass.int8, with accumulator data type `cutlass.int32` and epilogue compute data type `cutlass.float32` |
| LinearCombinationGeneric | $D  = activation(\alpha \times Accm + \beta \times C)$, available activations include `relu`, `leaky_relu`, `tanh`, `sigmoid`, `silu`, `hardswish`, and `gelu` |

The epilogue functors can be created as follows
```python
# LinearCombination
epilogue_functor = LinearCombination(
    element_C, alignment_c, element_acc, element_epilogue_compute
)

# LinearCombinationClamp
epilogue_functor = LinearCombinationClamp(
    element_C, alignment_c, element_acc, element_epilogue_compute
)

# FastLinearCombinationClamp
epilogue_functor = FastLinearCombinationClamp(
    element_C, alignment_c
)

# LinearCombinationGeneric
epilogue_functor = LinearCombinationGeneric(
    relu(element_epilogue_compute), element_C, alignment_c, 
    element_acc, element_epilogue_compute
)
```

We also provides an experimental feature "Epilogue Visitor Tree" for GEMM operation. The details can be found in [EpilogueVisitorTree](tools/library/scripts/pycutlass/docs/source/md/EpilogueVisitorTree.md).


### GEMM Operation

The GEMM Operation description can be created with 
```python
operation = GemmOperationUniversal(
    {compute_capability}, tile_description,
    A, B, C, epilogue_functor, 
    {swizzling_functor}, {visitor}
)
```
* `{compute_capability}` is an integer indicates the compute capability of the GPU. For A100, it is 80.
* `{swizzling_functor}` describes how threadblocks are scheduled on GPU. This is used to improve the L2 Locality ([detail](https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/)). Currently we support `cutlass.{IdentitySwizzle1|IdentitySwizzle2|IdentitySwizzle4|IdentitySwizzle8|BatchedIdentitySwizzle}`. The last one is used for batched or array GEMM.
* `{visitor}`: a bool variable indicates whether the epilogue visitor tree is used.

### GEMM Grouped Operation
The GEMM Grouped Operation description can be created with 
```python
operation = GemmOperationGrouped(
    compute_capability, tile_description,
    A, B, C, epilogue_functor, 
    swizzling_functor, {precompute_mode}
)
```
* `{precompute_mode}`: It could be `SchedulerMode.Host` or `SchedulerMode.Device`. See [examples/24_gemm_grouped](examples/24_gemm_grouped) for more details.


### Conv2d Operation
The Conv2d Operation description can be created with
```python
operation = Conv2dOperation(
    {conv_kind}, {iterator_algorithm},
    compute_capability, tile_description,
    A, B, C, {stride_support},
    epilogue_functor, swizzling_functor
)
```
* `{conv_kind}` defines which convolution is executed. Available options include `fprop`, `dgrad`, and `wgrad`.
* `{iterator_algorithm}` specifies the iterator algorithm used by the implicit GEMM in convolution. The options are as follows:
    * `analytic`: functionally correct in all cases but lower performance
    * `optimized`: optimized for R <= 32, S <= 32 and unity-stride dgrad
    * `fixed_channels`: analytic algorithm optimized for fixed channel count (C == AccessSize)
    * `few_channels`: Analytic algorithm optimized for few channels (C divisible by AccessSize)
* `{stride_support}`: distinguishes among partial specializations that accelerate certain problems where convolution
stride is unit.
    * `strided`: arbitrary convolution stride
    * `unity`: unit convolution stride

***
## Code Emission and Compilation
After implementing the operation description, the related host and device code can be compiled with
```python
import pycutlass

pycutlass.compiler.add_module([operation,])
```
Several operations can be compiled togather. The `nvcc` at `$CUDA_INSTALL_PATH/bin` is used by default as the compiler backend. But you can also switch to [CUDA Python](https://nvidia.github.io/cuda-python/overview.html)'s `nvrtc` with 
```python
pycutlass.compiler.nvrtc()
```
We also have an internal compiled artifact manager that caches the compiled kernel in both memory and disk. The `compiled_cache.db` at your workspace is the database that contains the binary files. You can delete the file if you want to recompile the kernels.
***
## Argument Processing
We provide argument wrapper to convert python tensors to the kernel parameters. Currently it supports [torch.Tensor](https://pytorch.org/), [numpy.ndarray](https://numpy.org/), and [cupy.ndarray](https://cupy.dev/). 
### GEMM Arguments
The Gemm arguments can be created with
```python
arguments = GemmArguments(
    operation=operation, problem_size={problem_size},
    A={tensor_A}, B={tensor_B}, C={tensor_C}, D={tensor_D},
    output_op={output_op},
    gemm_mode={gemm_mode},
    split_k_slices={split_k_slices}, batch={batch}
)
```
* `problem_size` is a `cutlass.gemm.GemmCoord(M, N, K)` object that defines $M\times N\times K$ matrix multiplication.
* `tensor_X`: user-provide tensors.
* `output_op`: the params for the epilogue functor.
* `gemm_mode`, `split_k_slices`, and `batch`:

|gemm_mode| split_k_slices | batch | remark|
|--|--|--|--|
|cutlass.gemm.Mode.Gemm | number of split-K slices | - | the ordinary GEMM or GEMM with serial split-K|
|cutlass.gemm.Mode.GemmSplitKParallel | number of split-K slices | - | GEMM Split-K Parallel|
|cutlass.gemm.Mode.Batched | - | batch size | Batched GEMM |
|cutlass.gemm.Mode.Array | - | batch size | Array GEMM |

### GEMM Grouped Arguments
The GEMM grouped arguments can be created with
```python
arguments = GemmGroupedArguments(
    operation, {problem_sizes_coord}, {tensor_As}, {tensor_Bs}, {tensor_Cs}, {tensor_Ds},
    output_op=output_op)
)
```
* `problem_size_coord` is a list of `cutlass.gemm.GemmCoord(M, N, K)` for each problem size.
* `tensor_Xs` is a list of user-provide tensors.
* `output_op`: the params of the epilogue functor

### Conv2d Arguments
The Conv2d arguments can be created with
```python
arguments = Conv2dArguments(
    operation, {problem_size}, {tensor_A},
    {tensor_B}, {tensor_C}, {tensor_D}, 
    {output_op}, 
    {split_k_mode},
    {split_k_slices}
)
```
* `problem_size`: it can be constructed with
    ```python
    problem_size = cutlass.conv.Conv2dProblemSize(
        cutlass.Tensor4DCoord(N, H, W, C),
        cutlass.Tensor4DCoord(K, R, S, C),
        cutlass.Tensor4DCoord(pad[0], pad[1], pad[2], pad[3]),
        cutlass.MatrixCoord(stride[0], stride[1]),
        cutlass.MatrixCoord(dilation[0], dilation[1]),
        cutlass.conv.Mode.cross_correlation, 
        split_k_slices, 1
    )
    ```
* `tensor_X` are user-provide tensors
* `output_op`: the params of the epilogue functor
* `split_k_mode`: currently we support `cutlass.conv.SplitKMode.Serial` and `cutlass.conv.SplitKMode.Parallel`.
* `split_k_slice`: number of split-k slices

For ordianry conv2d, just use `cutlass.conv.SplitKMode.Serial` with `split_k_slice=1`.

### Getting output_op
The way to create output_op is listed below
```python
output_op = operation.epilogue_type(*([alpha, beta] + args.activation_args)),
```
It is a list of arguments start with the scaling factor `alpha` and `beta`.
The `output_op` of EpilogueVisitorTree is slightly different. Please check [EpilogueVisitorTree](tools/library/scripts/pycutlass/docs/source/md/EpilogueVisitorTree.md) for details.


## Kernel Launching

With the arguments and operations, the kernel can be launched simply with
```python
operation.run(arguments)
```

## Sync results

We also provide function to synchronize the kernel execution. If you use `numpy`, it will also copy the result back to host. To do that, run
```python
arguments.sync()
```
If you use EpilogueVisitorTree, please call
```python
output_op.sync()
```

## Reduction Kernel behind Parallel Split-K

If you use parallel-split-K in GEMM or Conv2d, an additional reduction kernel is required. Please check [examples/40_cutlass_py](examples/40_cutlass_py) for detail.
