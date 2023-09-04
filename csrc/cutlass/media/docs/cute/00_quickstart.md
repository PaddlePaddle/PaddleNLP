# Getting Started With CuTe

CuTe is a collection of C++ CUDA template abstractions for defining and operating on hierarchically multidimensional layouts of threads and data. CuTe provides `Layout` and `Tensor` objects that compactly packages the type, shape, memory space, and layout of data, while performing the complicated indexing for the user. This lets programmers focus on the logical descriptions of their algorithms while CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design, implement, and modify all dense linear algebra operations.

The core abstraction of CuTe are the hierarchically multidimensional layouts which can be composed with data arrays to represent tensors. The representation of layouts is powerful enough to represent nearly everything we need to implement efficient dense linear algebra. Layouts can also be combined and manipulated via functional composition, on which we build a large set of common operations such as tiling and partitioning.

## System Requirements

CuTe shares CUTLASS 3.0's software requirements,
including NVCC with a C++17 host compiler.

## Knowledge prerequisites

CuTe is a CUDA C++ library.  It requires C++17
(the revision of the C++ Standard that was released in 2017).

Throughout this tutorial, we assume intermediate C++ experience.
For example, we assume that readers know
how to read and write templated functions and classes, and
how to use the `auto` keyword to deduce a function's return type.
We will be gentle with C++ and explain some things
that you might already know.

We also assume intermediate CUDA experience.
For example, readers must know
the difference between device and host code,
and how to launch kernels.

## Building Tests and Examples

CuTe's tests and examples build and run as part of CUTLASS's normal build process.
CuTe's unit tests live in the [`test/unit/cute`](../../../test/unit/cute) subdirectory.
Its examples live in the [`examples/cute`](../../../examples/cute) subdirectory.

## Library Organization

CuTe is a header-only C++ library, so there is no source code that needs building. Library headers are contained within the top level [`include/cute`](../../../include/cute) directory, with components of the library grouped by directories that represent their semantics.

|        Directory       |        Contents        |
|------------------------|------------------------|
| [`include/cute`](../../../include/cute) | Each header in the top level corresponds to one of the fundamental building blocks of CuTe, such as [`Layout`](../../../include/cute/layout.hpp) or [`Tensor`](../../../include/cute/tensor.hpp). |
| [`include/cute/container`](../../../include/cute/container) | Implementations of STL-like container objects, such as tuple, array, aligned array, and array views.  |
| [`include/cute/numeric`](../../../include/cute/numeric) | Templates that handle nonstandard floating-point types, unsigned integers, complex numbers, and integer sequence - like fundamental numeric data types.  |
| [`include/cute/algorithm`](../../../include/cute/algorithm) | Implementations of utility algorithms such as copy, fill, and clear that automatically leverage architecture-specific features if available. |
| [`include/cute/arch`](../../../include/cute/arch) | Wrappers for architecture-specific matrix-matrix multiply and copy instructions. |
| [`include/cute/atom`](../../../include/cute/atom) | Meta-information for instructions in `arch` and utilities like partitioning and tiling.

## Tutorial

This directory contains a CuTe tutorial in Markdown format.
The file
[`0x_gemm_tutorial.md`](./0x_gemm_tutorial.md)
explains how to implement dense matrix-matrix multiply using CuTe components.
It gives a broad overview of CuTe and thus would be a good place to start.

Other files in this directory discuss specific parts of CuTe.

* [`01_layout.md`](./01_layout.md) describes `Layout`, CuTe's core abstraction.

* [`02_layout_operations.md`](./02_layout_operations.md) describes more advanced `Layout` operations and the CuTe layout algebra.

* [`03_tensor.md`](./03_tensor.md) describes `Tensor`,
  a multidimensional array abstraction which composes `Layout`
  with an array of data.

* [`04_algorithms.md`](./04_algorithms.md) summarizes CuTe's
  generic algorithms that operate on `Tensor`s.

* [`0t_mma_atom.md`](./0t_mma_atom.md) demonstrates CuTe's meta-information and interface to our GPUs'
  architecture-specific Matrix Multiply-Accumulate (MMA) instructions.

* [`0x_gemm_tutorial.md`](./0x_gemm_tutorial.md) provides a walkthrough of building a GEMM from scratch using CuTe.

* [`0y_predication.md`](./0y_predication.md) explains what to do
  if a tiling doesn't fit evenly into a matrix.
