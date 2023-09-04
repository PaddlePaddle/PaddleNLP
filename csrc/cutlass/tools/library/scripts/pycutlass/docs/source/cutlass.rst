cutlass
=======

.. rubric:: Operator Classification

.. autoclass:: cutlass.OpClass
    :members:

.. rubric:: GEMM Layout

.. autoclass:: cutlass.RowMajor
    :members:

.. autoclass:: cutlass.ColumnMajor
    :members:

.. autoclass:: cutlass.RowMajorInterleaved32
    :members:

.. autoclass:: cutlass.ColumnMajorInterleaved32
    :members:

.. rubric:: Conv Layout

.. autoclass:: cutlass.TensorNHWC
    :members:

.. autoclass:: cutlass.TensorNC32HW32
    :members:

.. autoclass:: cutlass.TensorC32RSK32
    :members:

.. rubric:: Threadblock Swizzle

.. autoclass:: cutlass.dim3
    :special-members:
    :members:

.. autoclass:: cutlass.IdentitySwizzle1
    :special-members:
    :members:

.. autoclass:: cutlass.IdentitySwizzle2
    :special-members:
    :members:

.. autoclass:: cutlass.IdentitySwizzle4
    :special-members:
    :members:

.. autoclass:: cutlass.IdentitySwizzle8
    :special-members:
    :members:

.. autoclass:: cutlass.HorizontalSwizzle
    :special-members:
    :members:

.. autoclass:: cutlass.BatchedIdentitySwizzle
    :special-members:
    :members:

.. autoclass:: cutlass.StridedDgradIdentitySwizzle1
    :special-members:
    :members:

.. autoclass:: cutlass.StridedDgradIdentitySwizzle4
    :special-members:
    :members:

.. autoclass:: cutlass.StridedDgradHorizontalSwizzle
    :special-members:
    :members:

.. rubric:: Coordinates

.. autoclass:: cutlass.Tensor4DCoord
    :special-members:
    :members:

.. autoclass:: cutlass.Tensor3DCoord
    :special-members:
    :members:

.. autoclass:: cutlass.MatrixCoord
    :special-members:
    :members:


.. rubric:: Convolution

.. autoclass:: cutlass.conv.Operator
    :members:

.. autoclass:: cutlass.conv.IteratorAlgorithm
    :members:

.. autoclass:: cutlass.conv.StrideSupport
    :members:
