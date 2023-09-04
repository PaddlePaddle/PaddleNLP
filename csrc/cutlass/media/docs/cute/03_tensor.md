# CuTe Tensors

## A Tensor is a multidimensional array

CuTe's `Tensor` class represents a multidimensional array.
The array's elements can live in any kind of memory,
including global memory, shared memory, and register memory.

### Array access

Users access a `Tensor`'s elements in one of three ways:

* `operator()`, taking as many integral arguments as the number of modes,
  corresponding to the element's (possibly) multidimensional logical index;

* `operator()`, taking a `Coord` (an `IntTuple` of the logical indices); or

* `operator[]`, taking a `Coord` (an `IntTuple` of the logical indices).

### Slices: Get a Tensor accessing a subset of elements

Users can get a "slice" of a `Tensor`,
that is, a `Tensor` that accesses a subset of elements
of the original `Tensor`.

Slices happen through the same `operator()`
that they use for accessing an individual element.
Passing in `_` (the underscore character, an instance of `Underscore`)
has the same effect as `:` (the colon character) in Fortran or Matlab:
the resulting slice accesses all indices in that mode of the `Tensor`.

### Tensor's behavior determined by its Layout and Engine

A `Tensor`'s behavior is entirely determined by its two components,
which correspond to its two template parameters: `Engine`, and `Layout`.

For a description of `Layout`,
please refer to [the `Layout` section](./01_layout.md)
of this tutorial, or the [GEMM overview](./0x_gemm_tutorial.md).

An `Engine` represents a one-dimensional array of elements.
When users perform array access on a `Tensor`,
the `Tensor` uses its `Layout` to map from a logical coordinate
to a one-dimensional index.
Then, the `Tensor` uses its `Engine`
to map the one-dimensional index
to a reference to the element.
You can see this in `Tensor`'s implementation of array access.

```c++
decltype(auto) operator[](Coord const& coord) {
  return engine().begin()[layout()(coord)];
}
```

One could summarize almost all CuTe use cases as follows:

* create `Layout`s,

* create `Tensor`s with those `Layout`s, and

* invoke (either CuTe's, or custom) algorithms on those `Tensor`s.

### Ownership of the elements

`Tensor`s can be owning or nonowning.

"Owning" `Tensor`s behave like `std::array`.
When you copy the `Tensor`, you (deep-)copy its elements,
and the `Tensor`'s destructor deallocates the array of elements.

"Nonowning" `Tensor`'s behave like a (raw) pointer to the elements.
Copying the `Tensor` doesn't copy the elements,
and destroying the `Tensor` doesn't deallocate the array of elements.

Whether a `Tensor` is "owning" or "nonowning" depends entirely on its `Engine`.
This has implications for developers of generic `Tensor` algorithms.
For example, input `Tensor` parameters of a function
should be passed by const reference,
because passing the `Tensor`s by value
might make a deep copy of the `Tensor`'s elements.
It might also *not* make a deep copy of the elements;
there's no way to know without specializing the algorithm
on the `Tensor`'s `Engine` type.
Similarly, output or input/output `Tensor` parameters of a function
should be passed by (nonconst) reference.
Returning a `Tensor` might (or might not)
make a deep copy of the elements.

The various overloads of the `copy_if` algorithm in
[`include/cute/algorithm/copy.hpp`](../../../include/cute/algorithm/copy.hpp)
take their `src` (input, source of the copy) parameter
as `Tensor<SrcEngine, SrcLayout>& const`,
and take their `dst` (output, destination of the copy) parameter
as `Tensor<DstEngine, DstLayout>&`.
Additionally, there are overloads for mutable temporaries like
`Tensor<DstEngine, DstLayout>&&`
so that these algorithms can be applied directly to slices,
as in the following example.

```c++
copy(src_tensor(_,3), dst_tensor(2,_));
```

In C++ terms, each of the expressions
`src_tensor(_,3)`, and `dst_tensor(2,_)`
is in the "prvalue"
[value category](https://en.cppreference.com/w/cpp/language/value_category),
because it is a function call expression
whose return type is nonreference.
(In this case, calling `Tensor::operator()`
with at least one `_` (`Underscore`) argument
returns a `Tensor`.)
The prvalue `dst_tensor(2,_)` won't match
the `copy` overload taking
`Tensor<DstEngine, DstLayout>&`,
because prvalues can't be bound to
nonconst lvalue references (single `&`).
However, it will match the `copy` overload taking
`Tensor<DstEngine, DstLayout>&&`
(note the two `&&` instead of one `&`).
Calling the latter overload binds the reference
to the prvalue `dst_tensor(2,_)`.
This results in
[creation of a temporary](https://en.cppreference.com/w/cpp/language/implicit_conversion#Temporary_materialization)
`Tensor` result to be passed into `copy`.

### CuTe's provided `Engine` types

CuTe comes with three `Engine` types.

* `ArrayEngine<class T, int N>`: an owning `Engine`,
   representing an array of `N` elements of type `T`

* `ViewEngine<Iterator>`: a nonowning `Engine`,
  where `Iterator` is a random access iterator
  (either a pointer to an array, or something that acts like one)

* `ConstViewEngine<Iterator>`: a nonowning `Engine`,
  which is the view-of-const-elements version of `ViewEngine`

### "Tags" for different kinds of memory

`ViewEngine` and `ConstViewEngine` wrap pointers to various kinds of memory.
Users can "tag" the memory with its space -- e.g., global or shared --
by calling `make_gmem_ptr(g)` when `g` is a pointer to global memory,
or `make_smem_ptr(s)` when `s` is a pointer to shared memory.

Tagging memory makes it possible for CuTe's `Tensor` algorithms
to use the fastest implementation for the specific kind of memory.
It also avoids incorrect memory access.
For example, some kinds of optimized copy operations require
that the source of the copy be in global memory,
and the destination of the copy be in shared memory.
Tagging makes it possible for CuTe to dispatch
to those optimized copy operations where possible.
CuTe does this by specializing `Tensor` algorithms
on the `Tensor`'s `Engine` type.

### Engine members

In order for a type to be valid for use as an `Engine`,
it must have the following public members.

```c++
using value_type = /* ... the value type ... */;
using iterator =   /* ... the iterator type ... */;
iterator begin()   /* sometimes const */;
```

## Constructing a Tensor

### Nonowning view of existing memory

A `Tensor` can be a nonowning view of existing memory.
For this use case, users can create the `Tensor` by calling `make_tensor`
with two arguments: a wrapped pointer to the memory to view, and the `Layout`.
Users wrap the pointer by identifying its memory space:
e.g., global memory (via `make_gmem_ptr`) or shared memory (via `make_smem_ptr`).
`Tensor`s that view existing memory can have either static or dynamic `Layout`s.

Here are some examples of creating `Tensor`s
that are nonowning views of existing memory.

```c++
// Global memory (static or dynamic layouts)
Tensor gmem_8s     = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8d     = make_tensor(make_gmem_ptr(A), 8);
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{},16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape (      8  ,Int<16>{}),
                                                   make_stride(Int<16>{},Int< 1>{}));

// Shared memory (static or dynamic layouts)
Shape smem_shape = make_shape(Int<4>{},Int<8>{});
__shared__ T smem[decltype(size(smem_shape))::value];   // (static-only allocation)
Tensor smem_4x8_col = make_tensor(make_smem_ptr(&smem[0]), smem_shape);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(&smem[0]), smem_shape, GenRowMajor{});
```

### Owning array of register memory

A `Tensor` can also be an owning array of register memory.
For this use case, users can create the `Tensor`
by calling `make_tensor<T>(layout)`,
where `T` is the type of each element of the array,
and `layout` is the `Tensor`'s `Layout`.
Owning `Tensor`s must have a static `Layout`,
as CuTe does not perform dynamic memory allocation in `Tensor`s.

Here are some examples of creating owning `Tensor`s.

```c++
// Register memory (static layouts only)
Tensor rmem_4x8_col = make_tensor<float>(make_shape(Int<4>{},Int<8>{}));
Tensor rmem_4x8_row = make_tensor<float>(make_shape(Int<4>{},Int<8>{}), GenRowMajor{});
Tensor rmem_4x8_mix = make_tensor<float>(make_shape (Int<4>{},Int< 8>{}),
                                         make_stride(Int<2>{},Int<32>{}));
Tensor rmem_8   = make_fragment_like(gmem_8sx16d(_,0));
```

The `make_fragment_like` function makes an owning Tensor of register memory,
with the same shape as its input `Tensor` argument.

## Tensor use examples

### Copy rows of a matrix from global memory to registers

The following example copies rows of a matrix (with any `Layout`)
from global memory to register memory,
then executes some algorithm `do_something`
on the row that lives in register memory.

```c++
Tensor gmem = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{}, 16));
Tensor rmem = make_fragment_like(gmem(_, 0));
for (int j = 0; j < size<1>(gmem); ++j) {
  copy(gmem(_, j), rmem);
  do_something(rmem);
}
```

This code does not need to know anything the `Layout` of `gmem`
other than that it is rank-2 and that the first mode is a compile-time value.
The following code checks both of those conditions at compile time.

```c++
CUTE_STATIC_ASSERT_V(rank(gmem) == Int<2>{});
CUTE_STATIC_ASSERT_V(is_static<decltype(shape<0>(gmem))>{});
```

A `Tensor` encapsulates the data type, data location,
and possibly also the shape and stride of the tensor at compile time.
As a result, `copy` can dispatch, based on the types and Layouts of its arguments,
to use any of various synchronous or asynchronous hardware copy instructions
and can auto-vectorize the copy instructions in many cases as well.
CuTe's `copy` algorithm lives in
[`include/cute/algorithm/copy.hpp`](../../../include/cute/algorithm/copy.hpp).
For more details on the algorithms that CuTe provides,
please refer to [the algorithms section](./04_algorithms.md)
of the tutorial, or the
[CuTe overview in the GEMM tutorial](./0x_gemm_tutorial.md).

