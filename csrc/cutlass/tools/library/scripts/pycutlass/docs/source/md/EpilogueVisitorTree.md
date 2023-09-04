# Epilogue Visitor Tree
The Epilogue Visitor Tree is an experimental feature that directly generates epilogues from user-provide python functions.

## Usage

The Epilogue Visitor tree support many different operations. 

### Unary functions
Epilogue Visitor Tree supports unary functions like activation functions. For example,
```python
class UnaryEpilogue_(EpilogueVisitTree):
    def __call__(
        self, accum: 'tensor', c: 'tensor', 
        alpha: 'scalar', beta: 'scalar'):
        #
        T = leaky_relu.numpy(accum, 0.2)
        Z = alpha * T + beta * c
        return Z
epilogue_functor = UnaryEpilogue_(
    epilogue_functor, tile_description, math_inst.element_accumulator, 
    C.alignment, element_epilogue, C.element)
```

### Broadcast Operation
Epilogue Visitor Tree supports broadcasting row and column vectors to the whole output matrix. To use broadcast, you just need to specify whether the source vector is a `row` vector or a `column` vector. Here is an example.
```python
class ColumnBroadcast_(EpilogueVisitTree):
    def __call__(
        self, accum: 'tensor',  c: 'tensor', 
        vector: 'column', alpha: 'scalar', beta: 'scalar'):
        #
        T = accum + vector
        scale_T = leaky_relu.numpy(alpha * T, 0.2)
        Z = scale_T + beta * c
        return Z, T
epilogue_functor = ColumnBroadcast_(
    epilogue_functor, tile_description, math_inst.element_accumulator, 
    C.alignment, element_epilogue, C.element)
```

### Reduction Operation

Epilogue Visitor Tree also supports row and column-wise reduction in each threadblock tile. The syntax for reduction is
```python
{reduction_output} = reduction_op({input_tensor}, {row|column}, {Add}, {threadblock_shape.n|threadblock_shape.m})
```
The `{row|column}` indicates whether the `row` vectors are reduced or the `column` vectors are reduction. The `{Add}` specifies the reduction operation. The `{threadblock_shape.n|threadblock_shape.m}` are the reduction lengths.

**Constraint**
* The `{input_tensor}` can only be the name of source or intermediate result. `reduction_op(A + B, ...)` will not work, please use `C = A + B`, `reduction_op(C, ...)` instead.
* The `{reduction_output}` cannot be used in the epilogue. It will be directly written to global memory after the reduction is done.
```python
class RowReduction_(EpilogueVisitTree):
    def __call__(
        self, accum: 'tensor',  c: 'tensor', 
        alpha: 'scalar', beta: 'scalar'):
        #
        D = alpha * accum + tanh.numpy(beta * c)
        reduction = reduction_op(D, "row", "Add", args.threadblock_shape[1])
        return D, reduction
epilogue_functor = RowReduction_(
    epilogue_functor, tile_description, math_inst.element_accumulator, 
    C.alignment, element_epilogue, C.element)
epilogue_functor.initialize()
```

## Get output_op

As shown in the user guide, an `output_op` is required by the argument wrapper. We will take the `RowReduction_` as an example to show how to get `output_op`.
```python
class RowReduction_(EpilogueVisitTree):
    def __call__(
        self, accum: 'tensor',  c: 'tensor', 
        alpha: 'scalar', beta: 'scalar'):
        #
        D = alpha * accum + tanh.numpy(beta * c)
        reduction = reduction_op(D, "row", "Add", args.threadblock_shape[1])
        return D, reduction
epilogue_functor = RowReduction_(
    epilogue_functor, tile_description, math_inst.element_accumulator, 
    C.alignment, element_epilogue, C.element)
epilogue_functor.initialize()

cta_n = args.threadblock_shape[1]
num_cta_n = (problem_size.n() + cta_n - 1) // cta_n
reduction = np.zeros(shape=(args.batch * problem_size.m() * num_cta_n,), dtype=getattr(np, element_c))
# get output op
output_op = operation.epilogue_type(
    D=tensor_D, alpha=args.alpha, beta=args.beta, c=tensor_C, reduction=reduction, problem_size=[problem_size.m(), problem_size.n()]
)
```
Like other epilogue functors such as `LinearCombination`, the output op for EpilogueVisitorTree is also created with `operation.epilogue_type(*)`. However, there are two differences:
* The arguments need to be passed as keyword-arguments. The keywords are the argument names in `def __call__`.
* An additional `problem_size=[problem_size.m(), problem_size.n()]` is required. 


## Add new Unary Operation (e.g. Activation Function)
To add additional unary operation into epilogue visitor tree, a new unary op
should be created for `VisitorOpUnary`. We will take `tanh` as an example.

### Step 1: define TanhVisitor

The visitor defines the parameters and computation required by the unary option.
The unary operations are registered in [pycutlass/src/cpp/include/epilogue/epilogue_visitor_op/unary_ops.h](tools/library/scripts/pycutlass/src/cpp/include/epilogue/epilogue_visitor_op/unary_ops.h). But you can define it in any header file and include the header file in [pycutlass/src/cpp/include/epilogue/epilogue_visitor_op/visitor_op_unary.h](tools/library/scripts/pycutlass/src/cpp/include/epilogue/epilogue_visitor_op/visitor_op_unary.h).


* Two template arguments are required:
    * `T`: data type used to compute the unary operation
    * `N`: compute fragment length
* We also need to provide the `Arguments` and `Params` structures. The `Arguments` will be assembled by [ctypes](https://docs.python.org/3/library/ctypes.html), the `Params` will be generated from `Arguments` automatically. If the unary function takes no argument, an integer like `int tmp` can be provide to ensure the correctness of ctypes.
* The constructor can only take the `params` as the single argument.
* The operation is defined in `Array<T, N> operator()(Array<T, N> const &frag) const `. On common way to do that is first define a scalar computation, and them use it for the fragment computation with an unrolled for-loop.
* A guard function is required. If it returns `true`, it will disable all the children nodes of the unary node and return zeros to parent node. This is very helpful for multiplication with scalar while the scalar is `0`. For general cases, you can just return `true`. 
```c++
// T: data type used to compute the unary operation
// N: compute fragment length
template <typename T, int N>
struct TanhVisitor {
    /// Argument
    struct Arguments {
        // a placeholder argument to ensure correctness of ctypes
        int tmp;

        CUTLASS_HOST_DEVICE
        Arguments(): tmp(0) { };

        CUTLASS_HOST_DEVICE
        Arguments(int tmp): tmp(tmp) { };
    };

    /// Param
    struct Params {
        CUTLASS_HOST_DEVICE
        Params(){ };
        Params(Arguments const &args) { }
    };

    /// Constructor
    CUTLASS_HOST_DEVICE
    TanhVisitor(Params const &params) { }

    // scalar operator
    CUTLASS_HOST_DEVICE
    T tanh_op(T const &scalar) const {
        return fast_tanh(scalar);
    }

    /// vector operator
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        Array<T, N> y;

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < N; ++i) {
            y[i] = tanh_op(frag[i]);
        }

        return y;
    }

    // Guard
    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};
```

### Step 2: register Tanh function
After defining the function in C++, we need to register it in python. The class below gives an example.
* The init function takes the data type `element_compute`, which will be the `T` in the C++ template.
In the init function, we also generate the `_Arguments` class as a `ctypes.Structure`. It includes all the data members in the `TanhVisitor::Arguments`.
* The `_Arguments` need to be registered as `self.argument_type` of `tanh` class. 
* A `emit` function is required to emit the namespace and typename of `TanhVisitor`.
* A staticmethod as numpy reference is required to implement the python code to parse.

The built-in functions are defined in [pycutlass/src/pycutlass/epilogue.py](tools/library/scripts/pycutlass/src/pycutlass/epilogue.py). You can defined yours in any file as long as it can be found by [/pycutlass/src/pycutlass/parser.py](tools/library/scripts/pycutlass/src/pycutlass/parser.py).
```python
class tanh(ActivationFunctor):
    def __init__(self, element_compute) -> None:
        super().__init__()
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("tmp", ctypes.c_int)
            ]
            def __init__(self, *args) -> None:
                self.tmp = 0
        self.argument_type = _Arguments
    
    def emit(self):
        return "cutlass::TanhVisitor"

    @staticmethod
    def numpy(x: np.ndarray):
        return np.tanh(x)
```

### Step 3: Run the function
Now the new unary op is ready to use. An epilogue visitor tree can be built with
```python
class RowReduction_(EpilogueVisitTree):
    def __call__(
        self, accum: NDArray['tensor', 'float32'],  c: NDArray['tensor', 'float32'], 
        alpha: 'float32', beta: 'float32'):
        #
        D = alpha * accum + tanh.numpy(beta * c)
        reduction = reduction_op(D, "row", "Add", args.threadblock_shape[1])
        return D, reduction
epilogue_functor = RowReduction_(
    epilogue_functor, tile_description, math_inst.element_accumulator, 
    C.alignment, element_epilogue, C.element)
epilogue_functor.initialize()
```

## Limitations and Future work

Although the Epilogue Visitor Tree brings great flexibility to epilogue construction, as the epilogue is formulated as a single tree, there are several limitations.
* [Future Work] Serial and Parallel Split-K GEMM are not supported yet. 
    * To support serial split-k, additional tree transformation pass is required to inject a `binaryOpNode(Add)` + `TensorInputNode` before each `TensorOutputNode` to fetch the partial sum back. The `semaphore` also needs to be passed into epilogue. 
    * To support parallel split-k, an Reduction with visitor kernel is required.
* [Future Work] Convolution and GEMM Grouped are not supported yet.
    * To support Conv2d and GEMM Grouped, corresponding *_with_visitor kernels are required.

* [Limitation] If the same node is used by two operations (except that one of them is reduction), the node and all its offsprings will be executed twice.
* [Limitation] The result of reduction can only be used as the return value.
