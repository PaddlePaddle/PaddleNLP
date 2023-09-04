# CuTe dense matrix-matrix multiply tutorial

This section uses the CuTe functionality to write
a dense matrix-matrix multiply implementation.

## A simple dense matrix-matrix multiply example

In this section, we will go through
[this example](../../../examples/cute/tutorial/sgemm_nt_1.cu).
It illustrates a blocked GPU implementation of GEMM
that uses the building blocks of CuTe
to construct global and shared memory layout mappings
and partition threads among them.
This example is closest to the blocked GEMM
that a computer science student might be asked to implement
in a first-year graduate school
or upper-division undergraduate scientific computing course.

Readers who understand this section may also wish to study
CUTLASS's implementation of the stream-K GEMM algorithm,
which uses many features of CuTe.

### Filename and high-level interface

First, let's look at the example's filename `sgemm_nt_1.cu`.
"SGEMM" is the BLAS (Basic Linear Algebra Subroutines) abbreviation
for "Single-precision real, GEneral, Matrix-matrix Multiply."
(If we want to refer to matrix-matrix multiply for all data types,
we say "GEMM.")
The BLAS project started in the 1970s.
You can learn more about its history in Turing Award winner Jack Dongarra's
2004 Oral History interview by SIAM
(the Society for Industrial and Applied Mathematics),
and also in the C++ Standard document [P1417](https://wg21.link/p1417).
The abbreviation SGEMM unpacks as follows.

* "Single-precision" is Fortran-speak for float.
    The BLAS supports four different matrix or vector element types:

    * S for single precision (`float`),

    * D for double precision (`double`),

    * C for complex float (like C++'s `std::complex<float>`,
        where each of the real and imaginary components has type `float`),
        and

    * Z for complex double (like C++'s `std::complex<double>`).

* "GEneral" means that the matrix is represented
    as a two-dimensional dense array
    and not assumed to have any kind of symmetry.
    The BLAS supports a variety of matrix representations,
    including

    * SY: SYmmetric,

    * HE: HErmitian,

    * TR: TRiangular,

    * GB: General Banded,

    * SB: Symmetric Banded,

    * SP: Symmetric Packed, and

    * TP: Triangular Packed.

* MM means "Matrix-matrix multiply," as opposed to other operations,
    like MV (Matrix-Vector multiply).

The string "nt" in the filename means that
the first input matrix A is "Not transposed,"
while the second input matrix B is "Transposed."
That is, the function computes `C := beta * C + alpha * A * B^T`,
where the superscript T denotes the transpose of the matrix.
(We never change the input matrix in place or
store its entire transpose explicitly.
Instead, we reinterpret its data in place.)

GEMM's TRANSA and TRANSB arguments lets users specify
the transpose or Hermitian transpose (if complex)
of either or both input matrices A or B.
It turns out that implementations favor this "NT" case,
along with "TN" (A is Transposed, B is Not transposed).
We will explain why below.

As described, the original BLAS GEMM specifies
the dimensions of its matrices
as A is M x K, B is K x N, and C is M x N.
Out of convenience, CuTe interprets A
as M x K, B as N x K, and C as M x N. Instead of row-major or column-major (or Transposed
and Not-Transposed like above), we like to be more specific with M-major, N-major, or K-major.
Regardless, we'll still use the BLAS "NT" notation for high-level descriptions
of kernels when it's appropriate.

Now, let's look at the code.
We'll start with the kernel entry point `gemm_device`
at the top of the file.

```c++
template <class MShape, class NShape, class KShape,
          class TA, class AStride, class ABlockLayout, class AThreadLayout,
          class TB, class BStride, class BBlockLayout, class BThreadLayout,
          class TC, class CStride, class CBlockLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(MShape M, NShape N, KShape K,
            TA const* A, AStride dA, ABlockLayout blockA, AThreadLayout tA,
            TB const* B, BStride dB, BBlockLayout blockB, BThreadLayout tB,
            TC      * C, CStride dC, CBlockLayout       , CThreadLayout tC,
            Alpha alpha, Beta beta);
```

There are many template parameters;
we'll explain them all in due time.

`TA`, `TB`, and `TC` are the element types
of the matrices `A`, `B`, and `C`, respectively.
The two scalar constants `alpha` and `beta`
are part of what GEMM computes: `C = beta * C + alpha * A * B`.
Unlike the (traditional Fortran and C) BLAS,
CuTe lets you mix different matrix element types and/or scalar types.
The compiler will help, but it's somewhat up to you
to use types that are safe and efficient on the GPU.
For example, a custom arbitrary-precision real type
that does dynamic allocation inside may not work on the GPU at all.
Even if it does, it may not perform well.

This leaves five kinds of things to explain:

1. Shapes

2. Strides

3. Block layouts

4. Thread layouts

5. Launch bounds

### Shapes

The original Fortran BLAS GEMM lists the matrices' dimensions
in the order M, N, K. CuTe also uses this convention.
The "MShape" is just M,
the NShape is just N,
and the KShape is just K.
In this example, they are dynamic (run-time) values
defined at the top of the `gemm` host function
that invokes the device kernel.

```c++
// Define shapes (dynamic)
auto M = int(m);
auto N = int(n);
auto K = int(k);
```

Note that the function takes M, N, and K.
It doesn't take the shapes of the three matrices separately,
as (say) three different `Shape<int, int>` objects.
This is because matrix-matrix multiply constrains the shapes.

There's nothing mysterious about `int` here;
it's the usual C++ built-in integral type.
`auto M = int(m)` is a way to say
"convert `m` to an `int` if it's not already an `int`,
and assign it to the freshly declared variable `M`."
CuTe also has a capitalized `Int<Value>` templated type
for representing values as compile-time constants.
For example, `Int<5>` represents a compile-time `int` value 5.
(CuTe implements these as subclasses
of the C++ Standard Library class `std::integral_constant`.)
The above `gemm_device` function is templated on the types
of M, N, and K; this shows that CuTe can represent dimensions
as either run-time or compile-time values.

If you're familiar with the mdspan class going into C++23,
you might notice that CuTe represents shapes
a bit differently from mdspan.
mdspan uses `extents<class IndexType, size_t ... Extents>`
to represent a shape.
The `Extents` are zero or more compile-time values
(see below) representing the dimensions in the shape.
The `Extents...` are "non-type template parameters" (NTTPs) --
that is, they are not types, but compile-time values of type `size_t`.
If you use the special reserved `size_t` value `std::dynamic_extent`
as an extent value,
the resulting dimension is a run-time value
and is stored in the `extents` instance.
Any other extent value is a compile-time value
that is encoded in the extents type itself.
In contrast, CuTe represents a shape as `Shape<class ... Types>`.
The `Types...` are actual types, not NTTPs.
A built-in integral type like `int` or `uint64_t`
denotes a run-time dimension that is stored in the `Shape` instance,
while a compile-time value like `Int<5>`
encodes a compile-time dimension.
For example, the CuTe equivalent of
`extents<int, 3, dynamic_extent, 5>`
is `Shape<Int<3>, int, Int<5>>`.

#### Compile-time-ness of values

C++ values have three levels of "compile-time-ness":

1. dynamic (run-time) values,

2. constexpr values, and

3. static (compile-time) values.

(Rather than saying "C++ has,"
it's more accurate to say "C++17 has."
C++20 introduces `consteval` or "immediate" functions,
which make attempting to evaluate the function at run time
(any call not in an unevaluated context) a compiler error.
We'll ignore those for this tutorial,
since CuTe only requires C++17.)

The `constexpr` keyword was introduced in C++11.
It means something like
"the compiler can evaluate this expression at compile time."
It does NOT mean "the compiler MUST evaluate this at compile time."
If you use a `constexpr` expression in a `static_assert`
or as a non-type template argument,
then the compiler must evaluate the expression at compile time.
However, for `constexpr` occurring in other places,
the compiler may choose to store the value in registers or memory,
and/or do computations with the value at run time.
In some cases, the compiler must do that.
The following example shows that the compiler
might need to store `constexpr` values in memory sometimes.

```c++
// Some function defined in a different compilation unit.
extern int foo(int* x);

int bar()
{
  constexpr int value = 42; // a compile-time constant

  // Even constexpr variables have a sizeof,
  // because we still might need to take their address.
  static_assert(sizeof(value) == 4);

  // Compiler can't inspect foo to see how it uses the value,
  // so it has to store the value in some memory location
  // so that we can pass its address to the function.
  return foo(&value);
}
```

"Static" is an unfortunately overloaded term in C++.  Sometimes it means "the opposite of instance," like a "static function" or "static member" of a class.  (Some programming languages, like Java, say "class method" to refer to a "static function of a class.")  That's not what we mean here.  Instead, we mean "part of a compile-time type."  For example, `Int<1>` encodes the value 1 at compile time, as part of the type of a templated class `Int<Value>`.  `Int<3>` and `Int<4>` have different types.  You can get the value of of the type like this: `Int<3>::value`.  (The `value` is a `static constexpr` member of the class, where "static" means "opposite of instance.")  As soon as you go from `Int<3>` to `Int<3>::value`, you've gone from (3) above (a compile-time value) to (2) above (a `constexpr` value).  In some situations, this may mean that the compiler treats it as a run-time value.

#### Strides

We define a layout using both shapes and strides.
The shape just tells you the dimensions (modes, etc.) of the array.
The strides tell you the mapping from a multidimensional index
into a one-dimensional offset.
Here, we're describing the shapes and strides
of the "global" matrices A, B, and C.
The example defines the global matrices' strides
near the top of the `gemm` function.

```c++
// Define strides (mixed)
auto dA = make_stride(Int<1>{}, ldA); // (dM,dK)
auto dB = make_stride(Int<1>{}, ldB); // (dN,dK)
auto dC = make_stride(Int<1>{}, ldC); // (dM,dN)
```

To evaluate this mapping for a given multidimensional index, take the dot product of the indices with the strides.  For example, the offset of `A(index_m, index_k)` is `index_m * 1 + index_k * ldA`.  Note the implications for the compile-time-ness of the offset.  Any run-time value among either the shape or the strides makes the offset a run-time value.  Of course, if a particular stride is a compile-time constant (especially 1), it's easier for the compiler to optimize the arithmetic and result.

Note that in the original source code,
this example is missing the comments after each line.
We've added them in here,
as they stir a brief digression about shapes and modes.
The comment after B says (dN, dK), not (dK, dN).
This means that B is treated as an N x K matrix
instead of a K x N matrix.
As mentioned, CuTe follows the convention
that the meaning of matrix modes is
(M,K) for A, (N,K) for B, and (M,N) for C.
In particular, CuTe's convention is that
"the reduction mode is outermost."
The "reduction mode" of `Shape<M, N, K>` is K.
That's the mode over which we do a reduction,
that is, sum up products of matrix entries.
The K mode disappears in the output C.
"Outermost" here means "rightmost"
(literally, appearing rightmost in the list M, N, K).
Note that the shapes form a kind of Einstein tensor notation.
GEMM does Shape<M, N> = Shape<M, K> * Shape<K, N>.
In Einstein notation, the repeated index indicates
a sum of that term over all values of K.

We say in general that the leftmost mode is the "inner(most)" mode,
and the rightmost mode is the "outer(most)" mode.
This is because,
along with CuTe's convention of thinking of arrays as logically column major,
the leftmost mode is most commonly the mode with the most spatial locality.
It's very often the "most contiguous" mode.
For this reason, it's "the mode that we want in the innermost loop"
(in the nesting of loops that implements GEMM).
This is why we call it the "innermost" mode.
Its contiguity means that also call the innermost mode the "vector mode."

The vector mode also has special meaning:
it contains all of the information needed
to execute the smallest possible computation or communication operations
on hardware, that is, what CuTe calls the "atoms."

Modes are like units conceptually.
For example, you shouldn't mix M-mode indices with K-mode indices.
However, CuTe does nothing to enforce this.
(For example, CuTe does not require use of "tagged" index types.
Indexing works with the usual integer types.)

The previous paragraph relates to shapes, not strides.
Returning to the strides, the above code describes these strides as "mixed."
This means that they include both run-time and compile-time values.
For example, the stride between A(m, k) and A(m+1, k) is `Int<1>`,
a compile-time value 1.  The stride between A(m, k) and A(m, k+1),
however, is `ldA`, the "leading dimension of A," a run-time value.
The "leading dimension" of a matrix
refers to the stride between consecutive columns of a column-major matrix
(where the stride between consecutive rows is 1),
or the stride between consecutive rows of a row-major matrix
(where the stride between consecutive columns is 1).
This is a naming convention from the BLAS
and libraries that use it, like LAPACK.
For the purpose of this tutorial, it's just a naming convention
for "the stride that isn't the compile-time constant 1."

#### M-major, N-major, K-major

Note that we haven't uttered the phrases "column-major" or "row-major" here.  This is where the experience of a BLAS user diverges from the experience of a BLAS implementer.  BLAS users speak of "column-major" and "row-major" layouts.  C++23's `mdspan` class encodes these as `layout_left` resp. `layout_right`.  However, we don't speak of "column-major" or "row-major" in our GEMM implementations.

We say that a matrix is "M-major" if it is stride 1 in the M-mode, "N-major" if it is stride 1 in the N-mode, or "K-major" if it is stride 1 in the K-mode.  In the above code, A has shape (M, K) and strides (1, ldA).  Since A has stride 1 in the M mode, we say that A is "M major."  B has shape (N, K) and strides (1, ldB), so B is "N-major."  Similarly, C has shape (M, N) and strides (1, ldC), so C is "M major."

How do we translate this into the BLAS user's experience?
The following table illustrates for B and C.
(Throughout the table, "Impl" stands for "implementation.")

Note that the implementation reverses the order of B's modes,
and flips B's strides.
Recall that one evaluates a layout
by taking the dot product of the indices and strides.
Thus, reversing the order of both the modes and the strides
does not change this evaluation.

| Matrix | User's shape | User's layout | User's strides | Impl layout | Impl shape | Impl strides |
| ---    | ---          | ---           | ---            | ---         | ---        | ---          |
| C      | M x N        | Column major  | (1, LDC)       | M-major     | (M, N)     | (1, LDC)     |
| A      | M x K        | Column major  | (1, LDA)       | M-major     | (M, K)     | (1, LDA)     |

What about the matrix B?  We explained above that B is N-major.  How would that translate back into the BLAS user's experience?  We take a hint here from the filename including "nt."  The "nt" part of the name means that A is not transposed, while B is transposed.  The BLAS convention (see e.g., [the documentation for DGEMM](https://netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html)) is that if you take the transpose, then the dimensions refer to the transpose ("with op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix").  Thus, this example actually computes `C = beta * C + alpha * A * B^T`, where `B^T` is an K x N matrix with strides (LDB, 1).  The user's "original" matrix B is thus N x K, with strides (1, LDB) -- that's a column-major layout.  (Reversing the modes and the strides preserves the layout, since evaluating the layout mapping just takes the dot product of indices and strides.)  This lets us expand the above table to include B.

| Matrix | Transposed? | User's shape | User's layout | User's strides | Impl layout | Impl shape | Impl strides |
| ---    | ---         | ---          | ---           | ---            | ---         | ---        | ---          |
| C      | No          | M x N        | Column major  | (1, LDC)       | M-major     | (M, N)     | (1, LDC)     |
| A      | No          | M x K        | Column major  | (1, LDA)       | M-major     | (M, K)     | (1, LDA)     |
| B      | Yes         | N x K        | Column major  | (1, LDB)       | N-major     | (N, K)     | (1, LDB)     |

CuTe developers say: "In CuTe, you can't tell transposed
apart from non-transposed, MN-major from K-major, etc.
without inspecting the strides."
It's now a bit more clear what that means.
CuTe doesn't see whether A or B are transposed.
Instead, CuTe sees shapes and strides.
A CuTe developer must reason backwards from the shapes and strides
in order to see what the BLAS user sees.

Why does CuTe do this?  Consider that matrix multiply performs a reduction in the K-mode.  From the user's perspective, it's reducing across rows of the first input matrix, but across columns of the second input matrix.  If we instead mentally flip the modes of the first input matrix, then the implementation reduces over columns (the K mode) of both input matrices.  This leads to two cases in which the implementation can effectively treat both input matrices in the same way.  (If you call it with A and B reversed, it should even give the same results for these cases.)

| Case                       | User asks for A | User asks for B | Abbreviation |
| ---                        | ---             | ---             | ---          |
| A is M major, B is N major | Not transposed  | Transposed      | NT           |
| A and B are both K major   | Transposed      | Not transposed  | TN           |

This is why an introductory example starts with NT or TN.
For a summary of the four different transpose options for A and B,
and their corresponding implementation layouts,
please see the table below.

| Transpose abbreviation | User sees A transposed? | User sees B transposed? | A's impl layout | B's impl layout |
| ---                    | ---                     | ---                     | ---             | ---             |
| NT                     | No                      | Yes                     | M major         | N major         |
| TN                     | Yes                     | No                      | K major         | K major         |
| NN                     | No                      | No                      | M major         | K major         |
| TT                     | Yes                     | Yes                     | K major         | N major         |

#### MN-major and K-major

As we mentioned above, there are two "preferred arrangements," TN and NT.  In the TN arrangement, both A and B are K-major.  In the NT arrangement, A is M-major and B is N-major.  Even though the two stride-1 modes in NT have different names, it's still the leftmost mode for both A and B that has stride 1.  Thus, we can think of the NT arrangement as "MN-major," analogous to how the TN arrangement is "K-major."

The two preferred arrangements tend to work themselves into implementations, particularly when they use hardware instructions for accelerating matrix multiplies of blocks.  In some cases, the hardware instruction may require NT (MN-major) or TN (K-major).  For NN or TT, such instructions would require an intermediate transpose -- for example, when loading from global memory to shared memory.

### Block layouts

Efficient matrix multiply implementations loop over blocks.
For example, a typical GPU implementation strategy
is for each thread block to iterate over some number of blocks.
In the example, this loop occurs near the end of `gemm_device`.

```c++
// TUTORIAL: Example of a very simple compute loop
//   Data is read from global to shared memory via the tA|tB partitioning
//   gemm(.) operates on the shared memory directly via the tC partitioning

auto k_max = size<2>(tAgA);

for (int k = 0; k < k_max; ++k)
{
  // Copy A and B blocks from global memory to shared memory.
  copy(tAgA(_,_,k), tAsA);
  copy(tBgB(_,_,k), tBsB);

  // On some architectures, copy may be asynchronous.
  // This may call for extra synchronization instructions
  // beyond just __syncthreads().

  __syncthreads();

  // Compute gemm on shared memory input and register accumulator.
  // The "epilogue" after this loop will copy the accumulator
  // from the register file into global memory.
  gemm(tCsA, tCsB, tCrC);

  __syncthreads();
}
```

We will explain the notation in this loop below.  The important things to remember are that the coordinate `k` loops over the blocks which the calling thread is supposed to compute, the `copy` functions copy A resp. B blocks from global memory (the first argument) to shared memory (the second argument -- same as C++'s `std::copy`, but the opposite of `memcpy`), and the `gemm` function computes C += A * B on the shared memory blocks.

It turns out that copy takes an optional first argument, the "atom," as in the following.

```c++
copy(atom, source, destination);
```

The "atom" is metadata that explains how to do the copy operation.

There are a few topics to push onto the stack.

The copy function call shows a notation for taking slices of a tensor.  A CuTe `Tensor` is a multidimensional array view.  It consists of a pointer and a `Layout`.  You can learn more about `Tensor`s elsewhere in CuTe's documentation, but for now, please note that `tAgA(_,_,k)` means "create a Tensor that views (i, j, k) for all valid i, all valid j, and a specific value of k."  The result has rank one less than the original Tensor.  CuTe's underscore means the same thing as a single stand-alone colon in Fortran or Matlab.  Note also that CuTe uses the same notation for slices as for tensor indexing.  The implementation can distinguish the two cases by checking whether any of the arguments is an underscore.  In contrast, the C++23 class mdspan uses a separate function, `submdspan` (not in C++23, and proposed for C++26; see [P2630](https://wg21.link/p2630)), for slicing.

Fully understanding what `copy` and `gemm` do calls for learning about thread layouts as well, so we will wait to explain them completely.  For now, note that these functions are implicitly parallel, as they are called collectively by all threads in a thread block.

The block dimensions are defined near the top of the host function `gemm`.

```c++
// Define block sizes (static)
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<  8>{};
```

We see that these are fully compile-time dimensions.  This is often the case, especially when we use hardware instructions that only work for certain problem dimensions.  Three lines of code immediately below these construct the block layouts.

```c++
// Define the block layouts (static)
auto sA = make_layout(make_shape(bM,bK));
auto sB = make_layout(make_shape(bN,bK));
auto sC = make_layout(make_shape(bM,bN));
```

Here, the block layouts just come from the block dimensions.  A Layout has two things: a Shape, and Strides.  If the caller does not provide Strides, then CuTe computes Strides corresponding to the default "column-major" arrangement of data.  This just happens to match the global matrices' layouts, but in general doesn't have to.  For example, in the NN or TT cases, we may want to transpose one of the input matrices when copying from global memory to shared memory.

The example "comments out" some code that prints all the layouts on "thread 0" of each thread block.  If you enable the printing code and run the example, it will print all the layouts.  For example, sA prints as

```
sA
(_128,_8)
(_1,_128)
```

and sB prints as

```
sB
(_128,_8)
(_1,_128)
```

consistently with the definitions above.

If you have looked at other GEMM examples in CuTe, you might be wondering about hardware matrix-matrix multiply instructions.  Those instructions tend to require certain values for shapes and strides, that may be a function of the matrix's element type.  CuTe knows about these instructions and their required shapes and strides.  We will go into more detail about that elsewhere.

The `gemm_device` top-level kernel uses these block layouts to allocate shared memory buffers for A and B tiles.

```c++
// Shared memory buffers
__shared__ TA smemA[cosize_v<ABlockLayout>];
__shared__ TB smemB[cosize_v<BBlockLayout>];
```

Note how the shared memory buffers' sizes depend only on the A resp. B layouts (and element sizes). What's a `cosize_v`?  The "`_v`" is a C++ naming convention that specifies a function from one or more template argument(s), to a value.  In this case, it's a number of elements.  A layout is a function from a set of multidimensional coordinates to a set of one-dimensional array offsets.  It's a function, so we can speak of its domain and codomain.  The "cosize" of a layout is the size of its codomain.  (See e.g., CuTe's implementation of `Layout`.)  If we want to allocate a linear array, for which all the offsets produced by a layout are valid, then we can use the cosize of the layout as the length of the array (in terms of number of elements, not in terms of number of bytes).

### Thread layouts

CuTe uses a `Layout` to describe the assignment of threads to work items.
In this example, the host function `gemm` constructs the thread layouts
for A, B, and C.

```c++
// Define the thread layouts (static)
auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));
auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));
auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
```

That is, the thread layout for the A read is M-major 32x8, for the B read is N-major 32x8, and for the C compute/write is M-major 16x16. These thread layouts will partition the data for their respective stages.

#### The example uses compile-time thread and block layouts

Note that the device function `gemm_device` insists that all the thread and block layouts are static -- that is, known at compile time.  You can see this from the `CUTE_STATIC_ASSERT` statements near the top of `gemm_device`.  `CUTE_STATIC_ASSERT` is a wrapper for `static_assert`, which fails at compile time if its condition is `false`.

```c++
// Preconditions
CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);
```

Use of static layouts has two advantages.  First, it makes it easier to prove correctness of the algorithm.  If the code compiles, it's likely correct.  (On the other hand, new CuTe users may find themselves doing more debugging at compile time than they have before.)  Second, it makes it easier and faster for CuTe to dispatch to the correct optimized implementations (called "atoms" -- see below) for copying blocks and performing matrix multiplies.

#### The example's block gemm is parallel over elements of C

In the actual device function, `tC` has layout `CThreadLayout`.  You might recall that the kernel function `gemm_device` uses `CThreadLayout` to derive the launch bounds, specifically the maximum number of threads per block.  The launch bounds show up in the declaration of `gemm_device`.

```c++
template <class MShape, class NShape, class KShape,
          class TA, class AStride, class ABlockLayout, class AThreadLayout,
          class TB, class BStride, class BBlockLayout, class BThreadLayout,
          class TC, class CStride, class CBlockLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(MShape M, NShape N, KShape K,
            TA const* A, AStride dA, ABlockLayout blockA, AThreadLayout tA,
            TB const* B, BStride dB, BBlockLayout blockB, BThreadLayout tB,
            TC      * C, CStride dC, CBlockLayout       , CThreadLayout tC,
            Alpha alpha, Beta beta);
```

The "size" of `CThreadLayout` is the total number of threads, 16 * 16 = 256.  (We take `::value` because the size is actually `Int<256>`, a compile-time constant with a `static constexpr int value = 256` member.)  This suggests that the block gemm function (in the loop over blocks) parallelizes over elements of the C block.  We can see this as well from the kernel launch (at the end of the `gemm` host function), which uses the size of `CThreadLayout` as the block dimension.

```c++
// Define the thread layouts (static)
auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));
auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));
auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

dim3 dimBlock(size(tC));
dim3 dimGrid(ceil_div(size(M), size(bM)),
             ceil_div(size(N), size(bN)));
gemm_device
    <<< dimGrid, dimBlock, 0, stream >>>
    (M,  N,  K,
     A, dA, sA, tA,
     B, dB, sB, tB,
     C, dC, sC, tC,
     alpha, beta);
```

Note that dimBlock is single-dimensional (despite being a dim3), as the size of a layout is a single value.  We can see this also because the example only ever uses `threadIdx.x`, not `threadIdx.y`.  Yet, C's thread layout has shape (16, 16).  What's with that?  Recall that a thread layout maps from a "logical" coordinate space (possibly multidimensional tuples of indices) to (one-dimensional) integer indices.  In this case, `CThreadLayout` maps from pairs of indices in the Cartesian product space {0, 1, 2, ..., 15} x {0, 1, 2, ..., 15}, to one-dimensional indices 0, 1, 2, ..., 255.  The latter, the output of `CThreadLayout`, is the actual thread index `threadIdx.x` in this case.  `CThreadLayout` has only a shape (16, 16) and no nondefault strides, so it uses CuTe's default column-major arrangement (with strides (1, 16) in this case).

#### What does `local_tile` do?

The following code near the top of `gemm_device`
operates on the "global" (input and output) matrices A, B, and C
(where mA, mB, and mC are their Tensor representations).

```c++
// Get the appropriate blocks for this thread block --
// potential for thread block locality
auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB));  // (BLK_M,BLK_N,BLK_K)
auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)

Tensor gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, blk_shape, blk_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, blk_shape, blk_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

There are two new features here:

* `make_coord`, which returns a `Coord`, a multidimensional index which can be used as the input of a `Layout`; and

* `local_tile`, which we will explain below.

The `Coord`(inate) `blk_coord` refers to the set of blocks (indexed by k -- the underscore here indicating a free parameter) our thread block will access.  (The index k here doesn't mean the K mode; it's the same index as in the loop over blocks that does the computation.)

If we print out the `gA`, `gB`, and `gC` layouts, we get the following.

```
gA
(_128,_8,512)
(_1,5120,40960)

gB
(_128,_8,512)
(_1,5120,40960)

gC
(_128,_128)
(_1,5120)
```

All of these layouts come from the original input or output matrices A, B, and C.  Thus, they preserve the original strides, which are all the same in this example (when using default problem dimensions), 5120.  This is most easily seen in the gC layout.  For the other layouts, there is a clue in 5120 * 8 = 40960.  That is, every time we increase k by one, we "skip over 8 columns" of the global matrix, over to the next block of data.  This illustrates an important feature of CuTe, that it can view the same data with different modes and/or strides, as a way to identify parallelism or locality.

## Next steps

The above "simple GEMM" example's performance on many problems
is asymptotically optimal
with respect to the GPU's floating-point throughput.
Getting nearly peak performance
relative to the GPU's floating-point throughput,
for a wider variety of problem dimensions,
calls for more advanced techniques.
Please refer to other examples in this repository
to learn more about those techniques.
For example, the
[predication section of the tutorial](./0y_predication.md)
explains what to do if a matrix tiling
doesn't perfectly divide the matrix.

### Implement GEMM as generalized tensor constraction (GETT)

"GETT" here stands for "general(ized) tensor times tensor,"
a tensor contraction.

CuTe permits matrices to have nested `Layout`s.
For example, a matrix A can have a nested `Layout` for its M and N modes.
This means that we can use a "matrix" (`Tensor` with two modes)
to represent any `Tensor`.
This amounts to a "native hierarchical representation."

As a result, we can implement GETT by using
our existing GEMM implementation layers,
with a little bit of fancy custom predication for the K mode.
This is because the stride type of A
and the problem shape itself
are CuTe Shapes and Strides.
This lets us represent the hierarchical modes
of a tensor contraction problem
(which still fundamentally only have 4 modes --
batch mode,
two outer modes (one for A and one for B),
and one reduction mode --
each of which can now have as many nested modes as you want
for the contraction's inputs).
We thus implement GETT as contraction just in one mode -- the K mode.
However, K itself can be hierarchical and can have noncontiguous strides.
We can reorder the modes such that all contraction modes
become a single, possibly hierarchical K mode in the kernel.
This is how we would encode a contraction in multiple modes at once.
