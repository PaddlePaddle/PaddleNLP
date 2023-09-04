# Predication: What to do when tiling isn't perfect

The [GEMM tutorial](./0x_gemm_tutorial.md) shows how
we compute a matrix-matrix multiply
by iterating over tiles of the input matrices and output matrix.
The examples all assume that the tiles fit evenly into the matrices,
with no remainder.
What do we do if this is not the case?
For example, we might want to tile a 41 x 55 matrix into 4 x 8 tiles,
but 41 / 4 is 10 remainder 1, and 55 / 8 is 6 remainder 7.
What do we do with those "leftover" parts of the matrix?

Another way to say this, is that `logical_divide`
(CuTe's way of tiling layouts) "rounds up."
For example, if `N` is the layout (1000, 1) and `B` is the layout (128, 1),
then `logical_divide(N, B)` is the layout ((128, 8), (1, 128)).
This effectively rounds up the original shape N = 1000
into an 128 x 8 matrix (as if N = 1024).
What about those last 24 elements,
that aren't part of the original data?

The idiomatic CuTe way to solve this problem is through "predication."
Rather than trying to reason about the "remainder tiles,"
CuTe instead rounds up, but only tries to access data in each tile
that are part of the matrix.
This corresponds well with how our GPUs optimize:
branches without warp divergence are relatively fast.
It also matches the usual CUDA idiom
when dividing N work items in 1-D fashion over B thread blocks:
first test if "my thread" is out of bounds before doing work.

There are a few ways to figure out
which elements need to be predicated.
In-kernel GEMMs like to do this in the following way.

```c++
// Create the predicate tensor
Layout idA  = make_layout(shape(A));   // e.g. 1000:1
Layout idAB = logical_divide(idA, B);  // e.g. (128,8):(1,128)

Tensor pred = make_tensor<bool>(shape(idAB));
for (int i = 0; i < size(pred); ++i) {
  pred(i) = idAB(i) < size(A);
}

// ... intervening code ...

// Use the predicate tensor.  c is some coordinate.
// This code would likely live inside some algorithm.
if (pred(c)) { copy(idAB(c), smem(c)); }
```

The general procedure is that we

1. create an "identity" layout (`Layout idA  = make_layout(shape(A))`,
   in the above example) with the same shape as our original data;

2. repeat the same tiling/partitioning/slicing (possibly rounding up)
   on that identity layout (`Layout idAB = logical_divide(idA, B)`);

3. create a "predicate tensor" by comparing the coordinates
   of that reference layout with the bounds of the original layout;
   and then

4. use the predicate tensor to mask off accesses to out-of-bounds elements.

For example, suppose that we've partitioned A and B tiles
across threads as follows.

```c++
Tensor tAgA = local_partition(gA, tA, thread_idx);                  // (THR_M,THR_K,k)
Tensor tAsA = local_partition(sA, tA, thread_idx);                  // (THR_M,THR_K,PIPE)

Tensor tBgB = local_partition(gB, tB, thread_idx);                  // (THR_N,THR_K,k)
Tensor tBsB = local_partition(sB, tB, thread_idx);                  // (THR_N,THR_K,PIPE)
```

`tAgA` and `tBgB` partition the global A resp. B matrices over threads,
and `tAsA` and `tBsB` partition the shared memory tiles of A resp. B over threads.

The following code creates predicate tensors
corresponding to `tAgA` and `tBgB`.
They will be computed once in the prologue.
and will be used to mask off instructions in the inner loop.

```c++
Tensor tApA = make_tensor<bool>(make_shape (size<0>(tAgA), size<1>(tAgA)),
                                make_stride(     Int<1>{},      Int<0>{}));
Tensor tBpB = make_tensor<bool>(make_shape (size<0>(tBgB), size<1>(tBgB)),
                                make_stride(     Int<1>{},      Int<0>{}));
```

We're only thread-parallelizing over the leftmost (row) dimension,
so we only need to predicate over the leftmost dimension.
Thus, we can make the rightmost (column) stride zero,
since we will never actually address the rightmost dimension.

The following code creates "two-dimensional identity tensors"
that map coordinates (m,k) -> (m,k)
for the tile of data within the thread block.

```c++
Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));   // (BLK_M,BLK_K) -> (blk_m,blk_k)
Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));   // (BLK_N,BLK_K) -> (blk_n,blk_k)
```

The following lines then tile and partition
the two reference tensors
in exactly the same way the data were tiled and partitioned
into `tAsA` and `tBsB`.

```c++
Tensor tAcA = local_partition(cA, tA, thread_idx);
Tensor tBcB = local_partition(cB, tB, thread_idx);
```

Tiling and partitioning affect the offset and domain,
but not the codomain of the tensors,
so we're left with tensors that map `(thr_m,thr_k) -> (m,k)`
where `(thr_m,thr_k)` is this particular thread's subtensor of the tile
and `(m,k)` is the original codomain: a coordinate into the original tile.

The unrolled loops in the code below then compare
the m- and n-coordinates of those tensors with our known maximums
to mask off elements we are not allowed to access.

```c++
Tensor cA   = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
Tensor tAcA = local_partition(cA, tA, thread_idx);

Tensor cB   = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
Tensor tBcB = local_partition(cB, tB, thread_idx);

// Populate
CUTE_UNROLL
for (int m = 0; m < size<0>(tApA); ++m) {
  tApA(m,0) = get<0>(tAcA(m,0)) < m_max_coord;
}
CUTE_UNROLL
for (int n = 0; n < size<0>(tBpB); ++n) {
  tBpB(n,0) = get<0>(tBcB(n,0)) < n_max_coord;
}
```

Those last `for` loops fill in the two predicate tensors.
In this case, we only need to predicate over the leftmost dimension,
so we only address `(m,0)` resp. `(n,0)`.

We can then use the predicate tensors in `copy_if`
to copy only the elements for which the corresponding
predicate tensor elements are nonzero.

```c++
// Prefetch k_tile=0, gate these on k_residue as well
CUTE_UNROLL
for (int k = 0; k < size<1>(tAsA); ++k) {
  if (get<1>(tAcA(0,k)) >= -k_residue) { // some other condition on the column index
    copy_if(tApA, tAgA(_,k,0), tAsA(_,k,0));
  }
}

CUTE_UNROLL
for (int k = 0; k < size<1>(tBsB); ++k) {
  if (get<1>(tBcB(0,k)) >= -k_residue) { // some other condition on the column index
    copy_if(tBpB, tBgB(_,k,0), tBsB(_,k,0));
  }
}
```

Here are some advantages of this "reference tensor" approach.

1. It doesn't depend on the layout/strides of the tensor
   being predicated, just the logical bounds being imposed.

2. The partitioning stage can be anything.

3. It naturally extends to any-dimensional predication.

4. It's a natural generalization of a typical CUDA 1-D
   parallel vector access pattern,
   which computes an access index `k`
   (e.g., as `blockDim.x * blockIdx.x + threadIdx.x`)
   and then predicates access to the vector's `k`-th element
   on whether `k` is in bounds.

As an example of (3), the epilogue predication does exactly the same thing,

```c++
// Repeat with a tensor of coordinates for predication
Tensor cC   = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC)));
Tensor tCcC = thr_mma.partition_C(cC);

const bool isBetaZero = (beta == 0);

CUTE_UNROLL
for (int i = 0; i < size(tCrC); ++i) {
  if (elem_less(tCcC(i), make_coord(m_max_coord,n_max_coord))) {
    tCgC(i) = isBetaZero ? alpha * tCrC(i) : alpha * tCrC(i) + beta * tCgC(i);
  }
}
```

but with the mma responsible for the tiling/partitioning `tCcC`
so that the reference subtensor matches the accumulator's subtensor.
Then, the reference subtensor is predicated against the `if` bounds
(in both m- and n-coordinates) inside the `for` loop.

Another way to explain this is that we don't modify the tiles
to give you the "right" extents so that you never overrun.
Instead, we let you query the original coordinate
to see if that coordinate overruns.
This avoids all branching and variable/dynamic loop bounds
(thus maintaining load balance and synchronicity,
both very important in-kernel) in favor of predication.
It's also general enough to extend to all ranks,
all layouts of threads and data,
and all tiling/partitioning patterns.
