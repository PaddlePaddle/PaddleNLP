# CuTe Layouts

## Layout

This document describes `Layout`, CuTe's core abstraction.
A `Layout` maps from (a) logical coordinate space(s)
to a physical index space.

`Layout`s present a common interface to multidimensional array access
that abstracts away the details of how the array's elements are organized in memory.
This lets users write algorithms that access multidimensional arrays generically,
so that layouts can change, without users' code needing to change.

CuTe also provides an "algebra of `Layout`s."
`Layout`s can be combined and manipulated
to construct more complicated layouts
and to partition them across other layouts.
This can help users do things like partition layouts of data over layouts of threads.

## Layouts and Tensors

Any of the `Layout`s discussed in this section can be composed with data -- a pointer or an array -- to create a `Tensor`. The responsibility of the `Layout` is to define valid coordinate space(s) and, therefore, the logical shape of the data and map those into an index space. The index space is precisely the offset that would be used to index into the array of data.

For details on `Tensor`, please refer to the
[`Tensor` section of the tutorial](./03_tensor.md).

## Shapes and Strides

A `Layout` is a pair of `Shape` and `Stride`.
Both `Shape` and `Stride` are `IntTuple` types.

### IntTuple

An `IntTuple` is an integer or a tuple of `IntTuple`s.
This means that `IntTuple`s can be arbitrarily nested.
Operations defined on `IntTuple`s include the following.

* `get<I>(IntTuple)`: The `I`th element of the `IntTuple`. Note that `get<0>` is defined for integer `IntTuples`.

* `rank(IntTuple)`: The number of elements in an `IntTuple`. An int has rank 1, a tuple has rank `tuple_size`.

* `depth(IntTuple)`: The number of hierarchical `IntTuple`s. An int has depth 0, a tuple has depth 1, a tuple that contains a tuple has depth 2, etc.

* `size(IntTuple)`: The product of all elements of the IntTuple.

We write `IntTuple`s with parenthesis to denote the hierarchy. E.g. `6`, `(2)`, `(4,3)`, `(3,(6,2),8)` are all `IntTuple`s.

## Layout

A `Layout` is then a pair of `IntTuple`s. The first defines the abstract *shape* of the layout and the second defines the *strides*, which map from coordinates within the shape to the index space.

As a pair of `IntTuple`s, we can define many similar operations on `Layout`s including

* `get<I>(Layout)`: The `I`th sub-layout of the `Layout`.

* `rank(Layout)`: The number of modes in a `Layout`.

* `depth(Layout)`: The number of hierarchical `Layout`s. An int has depth 0, a tuple has depth 1, a tuple that contains a tuple has depth 2, etc.

* `shape(Layout)`: The shape of the `Layout`.

* `stride(Layout)`: The stride of the `Layout`.

* `size(Layout)`: The logical extent of the `Layout`. Equivalent to `size(shape(Layout))`.

### Hierarchical access functions

`IntTuple`s and thus `Layout`s can be arbitrarily nested.
For convenience, we define versions of some of the above functions
that take a sequence of integers, instead of just one integer.
This makes it possible to access elements
inside of nested `IntTuple` or `Layout`.
For example, we permit `get<I...>(x)`, where `I...` here
and throughout this section is a "C++ parameter pack"
that denotes zero or more (integer) template arguments.
That is, `get<I0,I1,...,IN>(x)` is equivalent to
`get<IN>(` $\dots$ `(get<I1>(get<I0>(x)))` $\dots$ `))`,
where the ellipses are pseudocode and not actual C++ syntax.
These hierarchical access functions include the following.

* `rank<I...>(x)  := rank(get<I...>(x))`. The rank of the `I...`th element of `x`.

* `depth<I...>(x) := depth(get<I...>(x))`. The depth of the `I...`th element of `x`.

* `size<I...>(x)  := size(get<I...>(x))`. The size of the `I...`th element of `x`.

### Vector examples

Then, we can define a vector as any `Shape` and `Stride` pair with `rank == 1`.
For example, the `Layout`

```
Shape:  (8)
Stride: (1)
```

defines a contiguous 8-element vector.
Similarly, with a stride of `(2)`,
the interpretation is that the eight elements
are stored at positions 0, 2, 4, $\dots$.

By the above definition, we *also* interpret

```
Shape:  ((4,2))
Stride: ((1,4))
```

as a vector, since its shape is rank 1. The inner shape describes a 4x2 layout of data in column-major order, but the extra pair of parenthesis suggest we can interpret those two modes as a single 1-D 8-element vector instead. Due to the strides, the elements are also contiguous.

### Matrix examples

Generalizing, we define a matrix as any `Shape` and `Stride` pair with rank 2. For example,

```
Shape:  (4,2)
Stride: (1,4)
  0   4
  1   5
  2   6
  3   7
```

is a 4x2 column-major matrix, and

```
Shape:  (4,2)
Stride: (2,1)
  0   1
  2   3
  4   5
  6   7
```

is a 4x2 row-major matrix.

Each of the modes of the matrix can also be split into *multi-indices* like the vector example.
This lets us express more layouts beyond just row major and column major. For example,

```
Shape:  ((2,2),2)
Stride: ((4,1),2)
  0   2
  4   6
  1   3
  5   7
```

is also logically 4x2, with a stride of 2 across the rows but a multi-stride down the columns.
Since this layout is logically 4x2,
like the column-major and row-major examples above,
we can _still_ use 2-D coordinates to index into it.

## Constructing a `Layout`

A `Layout` can be constructed in many different ways.
It can include any combination of compile-time (static) integers
or run-time (dynamic) integers.

```c++
auto layout_8s = make_layout(Int<8>{});
auto layout_8d = make_layout(8);

auto layout_2sx4s = make_layout(make_shape(Int<2>{},Int<4>{}));
auto layout_2sx4d = make_layout(make_shape(Int<2>{},4));

auto layout_2x4 = make_layout(make_shape (2, make_shape (2,2)),
                              make_stride(4, make_stride(1,2)));
```

## Using a `Layout`

The fundamental use of a `Layout` is to map between logical coordinate space(s) and index space. For example, to print an arbitrary rank-2 layout, we can write the function

```c++
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}
```

which produces the following output for the above examples.

```
> print2D(layout_2sx4s)
  0   2   4   6
  1   3   5   7
> print2D(layout_2sx4d)
  0   2   4   6
  1   3   5   7
> print2D(layout_2x4)
  0   2   1   3
  4   6   5   7
```

The multi-indices within the `layout_4x4` example are handled as expected and interpreted as a rank-2 layout.

Note that for `layout_1x4`, we're using a 1-D coordinate for a 2-D multi-index in the second mode. In fact, we can generalize this and treat all of the above layouts as 1-D layouts.  For instance, the following `print1D` function

```c++
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
```

produces the following output for the above examples.

```
> print1D(layout_8s)
  0   1   2   3   4   5   6   7
> print1D(layout_8d)
  0   1   2   3   4   5   6   7
> print1D(layout_2sx4s)
  0   1   2   3   4   5   6   7
> print1D(layout_2sx4d)
  0   1   2   3   4   5   6   7
> print1D(layout_2x4)
  0   4   2   6   1   5   3   7
```

This shows explicitly that all of the layouts are simply folded views of an 8-element array.

## Summary

* The `Shape` of a `Layout` defines its coordinate space(s).

    * Every `Layout` has a 1-D coordinate space.
      This can be used to iterate in a "generalized-column-major" order.

    * Every `Layout` has a R-D coordinate space,
      where R is the rank of the layout.
      These spaces are ordered _colexicographically_
      (reading right to left, instead of "lexicographically,"
      which reads left to right).
      The enumeration of that order
      corresponds to the 1-D coordinates above.

    * Every `Layout` has an h-D coordinate space where h is "hierarchical." These are ordered colexicographically and the enumeration of that order corresponds to the 1-D coordinates above. An h-D coordinate is congruent to the `Shape` so that each element of the coordinate has a corresponding element of the `Shape`.

* The `Stride` of a `Layout` maps coordinates to indices.

    * In general, this could be any function from 1-D coordinates (integers) to indices (integers).

    * In `CuTe` we use an inner product of the h-D coordinates with the `Stride` elements.
