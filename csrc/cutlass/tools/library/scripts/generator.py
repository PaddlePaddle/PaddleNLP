#
# \file generator.py
#
# \brief Generates the CUTLASS Library's instances
#

import enum
import os.path
import shutil
import argparse
import logging

from library import *
from manifest import *
from itertools import product

###################################################################################################

#
def CudaToolkitVersionSatisfies(semantic_ver_string, major, minor, patch = 0):

  # by default, use the latest CUDA Toolkit version
  cuda_version = [11, 0, 132]

  # Update cuda_version based on parsed string
  if semantic_ver_string != '':
    for i, x in enumerate([int(x) for x in semantic_ver_string.split('.')]):
      if i < len(cuda_version):
        cuda_version[i] = x
      else:
        cuda_version.append(x)
  return cuda_version >= [major, minor, patch]


###################################################################################################
###################################################################################################

#
def EpilogueAlignment(max_alignment, tile, epilogue_steps = 8):
  ''' Helper to compute the maximum alignment of the epilogue '''

  def product(X, identity = 1):
    result = identity
    for item in X:
      result *= item
    return result

  elements_per_thread = product(tile.threadblock_shape[:-1]) // product(tile.warp_count) // 32 // epilogue_steps
  return min(max_alignment, elements_per_thread)

def DefaultSwizzlingFunctor():
    return SwizzlingFunctor.Identity8;
    # To use StreamK decomposition for basic GEMMs, set `swizzling_functor = SwizzlingFunctor.StreamK`

#
def CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = DefaultSwizzlingFunctor()):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations


# Generates 3.0 API based GemmUniversal API kernels. Alignment constraits are folded in with layouts
def CreateGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_type,
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none), ]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0]]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for complex_transform in complex_transforms:
        A = TensorDescription(
            element_a, layout[0][0], layout[0][1], complex_transform[0])
        B = TensorDescription(
            element_b, layout[1][0], layout[1][1], complex_transform[1])
        C = TensorDescription(element_c, layout[2][0], layout[2][1])

        operation = GemmOperation(
            GemmKind.Universal3x, tile_description.minimum_compute_capability,
            tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

        manifest.append(operation)
        operations.append(operation)

  return operations

#
def CreateSparseGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.Sparse]
  
  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:
  
            alignment_c = min(8, alignment)
 
            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Sparse, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

#
def CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.PlanarComplex, GemmKind.PlanarComplexArray]
  
  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]
  
  for gemm_kind in gemm_kinds:
    for layout in layouts:
      for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
          for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            manifest.append(GemmOperation(gemm_kind, \
              tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue))
  return

#
def CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GroupedGemmOperation(GemmKind.Grouped, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

#
def CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, data_type, \
  alignment_constraints, blas_mode, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  element_a, element_c, element_epilogue = data_type
  
  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for fill_mode in fill_modes:
      for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
          
          # SERK supported layouts (RowMajor, ColumnMajor) with no conjugation
          complex_transform = ComplexTransform.none

          # HERK supported layouts (RowMajor + conj, ColumnMajor)
          if blas_mode == BlasMode.hermitian and layout[0] == LayoutType.RowMajor:
            complex_transform = ComplexTransform.conj

          alignment_c = 1 # Alignment only applies to A in SYRK
   
          A = TensorDescription(element_a, layout[0], alignment, complex_transform)
          C = SymmetricTensorDescription(element_c, layout[1], fill_mode, alignment_c)

          # Rank-K update
          new_operation = RankKOperation(RankKKind.Universal, tile_description.minimum_compute_capability, \
            tile_description, A, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

          manifest.append(new_operation)
          operations.append(new_operation)
          
          # Rank-2K update
          new_operation = Rank2KOperation(RankKKind.Universal, tile_description.minimum_compute_capability, \
            tile_description, A, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

#
def CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type
  
  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for side_mode in side_modes:
      for fill_mode in fill_modes:
        for diag_type in diag_types:
          for tile_description in tile_descriptions:
            for alignment in alignment_constraints:
              for complex_transform in complex_transforms:
        
                  alignment_c = min(8, alignment)
       
                  A = TriangularTensorDescription(element_a, layout[0], side_mode, fill_mode, diag_type,
                                                  alignment, complex_transform)
                  B = TensorDescription(element_b, layout[1], alignment)
                  C = TensorDescription(element_c, layout[2], alignment_c)

                  new_operation = TrmmOperation(TrmmKind.Universal, tile_description.minimum_compute_capability, \
                    tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

                  manifest.append(new_operation)
                  operations.append(new_operation)

  return operations

#
def CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, data_type, \
  alignment_constraints, blas_mode, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  element_a, element_b, element_c, element_epilogue = data_type
  
  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for side_mode in side_modes:
      for fill_mode in fill_modes:
        for tile_description in tile_descriptions:
          for alignment in alignment_constraints:
            
            # SYMM supported layouts (RowMajor, ColumnMajor) with no conjugation
            complex_transform = ComplexTransform.none

            alignment_a = 1 # No vectorized access for the triangular matrix
            alignment_c = min(8, alignment)
       
            A = SymmetricTensorDescription(element_a, layout[0], fill_mode, alignment_a, complex_transform, side_mode)
            # tensor A and B have same data type and layout
            B = TensorDescription(element_b, layout[0], alignment)
            C = TensorDescription(element_c, layout[1], alignment_c)

            # SYMM/HEMM update
            new_operation = SymmOperation(SymmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

            manifest.append(new_operation)
            operations.append(new_operation)
            
            # SYMM/HEMM update
            new_operation = SymmOperation(SymmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor, blas_mode)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

###########################################################################################################
#   ConvolutionOperator support variations
#        ____________________________________________________________________
#         ConvolutionalOperator |      Analytic          |    Optimized
#        ____________________________________________________________________
#        |       Fprop          |     (strided)          |    (strided)
#        |       Dgrad          |     (strided, unity*)  |    (strided, unity)
#        |       Wgrad          |     (strided)          |    (strided)
#        ____________________________________________________________________
#
# Note :  Operator marked (*) are supported but not generated to keep the instantiated kernel count low
###########################################################################################################
# Convolution for 2D operations
def CreateConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):
  
  element_a, element_b, element_c, element_epilogue = data_type
  
  # one exceptional case
  
  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]
    iterator_algorithms = [IteratorAlgorithm.Optimized]

  operations = []

  for tile in tile_descriptions:
    for alignment in alignment_constraints:

      alignment_c = min(8, alignment)

      A = TensorDescription(element_a, layout[0], alignment)
      B = TensorDescription(element_b, layout[1], alignment)
      C = TensorDescription(element_c, layout[2], alignment_c)
      
      swizzling_functor_ = swizzling_functor
  
      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:
  
        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operations = [
            # None grouped kernel
            Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
              A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_),
          ]

          # Instance group conv kernel
          if tile.math_instruction.opcode_class == OpcodeClass.TensorOp and A.layout == LayoutType.TensorNHWC and \
            tile.minimum_compute_capability >= 80:
            # SingleGroup kernel
            new_operations.append(Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
              A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_, group_mode=GroupMode.SingleGroup))

            # Analytic iterator supports MultipleGroup mode
            if iterator_algorithm == IteratorAlgorithm.Analytic:
              new_operations.append(Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
                A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_, group_mode=GroupMode.MultipleGroup))

          for new_operation in new_operations:
            manifest.append(new_operation)
            operations.append(new_operation)
  
      #
      # Conv2d Dgrad
      #
      if ConvKind.Dgrad in conv_kinds:
  
        # Unity stride for Analytic and Optimized Dgrad
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor, swizzling_functor_)
  
          manifest.append(new_operation)
          operations.append(new_operation)
  
        # Strided support for Analytic Dgrad
        # strided dgrad uses a special threadblock swizzle
        # note that SwizzlingFunctor.StridedDgradHorizontal might be 
        # better for problem sizes with large activation channel count
        swizzling_functor_strided_dgrad_ = SwizzlingFunctor.StridedDgradIdentity1
  
        if IteratorAlgorithm.Analytic in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, IteratorAlgorithm.Analytic, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_strided_dgrad_)
  
          manifest.append(new_operation)
          operations.append(new_operation)
        
        # Strided support for Optimized Dgrad
        if IteratorAlgorithm.Optimized in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, IteratorAlgorithm.Optimized, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_strided_dgrad_)
  
          manifest.append(new_operation)
          operations.append(new_operation)
  
      #
      # Conv2d Wgrad
      #
      if ConvKind.Wgrad in conv_kinds:
       
        # Strided support for Analytic and Optimized Wgrad
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Wgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)
  
          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 2D operations specialized for few channels
def CreateConv2dFixedChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.FixedChannels,]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    channel_counts = [channel_counts[0],]

  operations = []



  for tile in tile_descriptions:
    for channel_count in channel_counts:

      alignment_c = EpilogueAlignment(channel_count, tile)

      A = TensorDescription(element_a, layout[0], channel_count)
      B = TensorDescription(element_b, layout[1], channel_count)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 2D operations specialized for few channels
def CreateConv2dFewChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.FewChannels,]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    channel_counts = [channel_counts[0],]

  operations = []

  for tile in tile_descriptions:
    for channel_count in channel_counts:

      alignment_c = EpilogueAlignment(channel_count, tile)

      A = TensorDescription(element_a, layout[0], channel_count)
      B = TensorDescription(element_b, layout[1], channel_count)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 3D operations
def CreateConv3dOperator(manifest, layout, tile_descriptions, data_type, alignment, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], epilogue_functor = EpilogueFunctor.LinearCombination):
  
  element_a, element_b, element_c, element_epilogue = data_type
  
  # one exceptional case
  alignment_c = min(8, alignment)
  
  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size and optimized iterators
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    iterator_algorithms = [IteratorAlgorithm.Optimized]

  operations = []

  # All tile sizes for Conv3dFprop and Conv3dWgrad
  for tile in tile_descriptions:
    A = TensorDescription(element_a, layout, alignment)
    B = TensorDescription(element_b, layout, alignment)
    C = TensorDescription(element_c, layout, alignment_c)
    
    #
    # Conv3d Fprop
    #
    if ConvKind.Fprop in conv_kinds:
      # Strided support for Analytic and Optimized Fprop
      for iterator_algorithm in iterator_algorithms:
        new_operation = Conv3dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
                                        A, B, C, element_epilogue, StrideSupport.Strided)
        manifest.append(new_operation)
        operations.append(new_operation)
    #
    # Conv3d Wgrad
    #
    if ConvKind.Wgrad in conv_kinds:
     
      # Strided support for Analytic and Optimized Wgrad
      for iterator_algorithm in iterator_algorithms:
        new_operation = Conv3dOperation(ConvKind.Wgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
          A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)
        manifest.append(new_operation)
        operations.append(new_operation)

  # All tile sizes for Conv3dDgrad
  for tile in tile_descriptions:
    
    A = TensorDescription(element_a, layout, alignment)
    B = TensorDescription(element_b, layout, alignment)
    C = TensorDescription(element_c, layout, alignment_c)
    
    #
    # Conv3d Dgrad
    #
    if ConvKind.Dgrad in conv_kinds:
      # Unity stride for Optimized Dgrad
      new_operation = Conv3dOperation(ConvKind.Dgrad, IteratorAlgorithm.Optimized, tile.minimum_compute_capability, tile,\
        A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor)
      
      manifest.append(new_operation)
      operations.append(new_operation)
      
      # Strided support for Analytic Dgrad 
      # Conv3dDgrad has a naive strided support which does not cut down redundant MMAs   
      new_operation = Conv3dOperation(ConvKind.Dgrad, IteratorAlgorithm.Analytic, tile.minimum_compute_capability, tile,\
        A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)
      
      manifest.append(new_operation)
      operations.append(new_operation)

  return operations

# Convolution for Depthwise 2d conv
def CreateDepthwiseConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):
  
  element_a, element_b, element_c, element_epilogue = data_type
  
  # iterator algorithm (FixedStrideDilation, Optimized)
  iterator_algorithms = [IteratorAlgorithm.FixedStrideDilation, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  operations = []

  for tile in tile_descriptions:
    for alignment in alignment_constraints:

      alignment_c = min(8, alignment)

      A = TensorDescription(element_a, layout[0], alignment)
      B = TensorDescription(element_b, layout[1], alignment)
      C = TensorDescription(element_c, layout[2], alignment_c)
      
      swizzling_functor_ = swizzling_functor

      if ConvKind.Fprop in conv_kinds:
  
        # Strided support for Optimized and FixedStridedDilation Depthwise Conv
        for iterator_algorithm in iterator_algorithms:
          stride_support = StrideSupport.Strided
          if iterator_algorithm == IteratorAlgorithm.FixedStrideDilation:
              if tile.stride == [-1, -1] or tile.dilation == [-1,-1]:
                continue
              stride_support = StrideSupport.Fixed

          if iterator_algorithm == IteratorAlgorithm.Optimized:
              if tile.stride != [-1, -1] or tile.dilation != [-1,-1]:
                continue 
          new_operation = Conv2dOperation(ConvKind.Fprop, 
                                          iterator_algorithm, 
                                          tile.minimum_compute_capability, 
                                          tile,
                                          A, B, C, 
                                          element_epilogue, 
                                          stride_support, 
                                          epilogue_functor, 
                                          swizzling_functor_, 
                                          group_mode=GroupMode.Depthwise)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

###################################################################################################
###################################################################################################

#
def GenerateSM50_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f64, DataType.f64, DataType.f64,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 50
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    if math_inst.element_a == DataType.f32:
      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM50_Simt_complex(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 50
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
    ]

    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM50(manifest, cuda_version):
  GenerateSM50_Simt(manifest, cuda_version)
  GenerateSM50_Simt_complex(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM60_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 60
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#
def GenerateSM60_Simt_DepthwiseConv2d(manifest, cuda_version):

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]
  
  min_cc = 60
  max_cc = 1024

  alignment_constraints = [8,]

  filter_3x3 = [3, 3]
  filter_5x5 = [5, 5]

  # [stride_h, stride_w]
  # [-1, -1] means all stride size.
  strides = [[-1,-1], [1, 1], [2, 2]]
  # [dilation_h, dilation_w]
  # [-1, -1] means all dilation size.
  dilations = [[-1,-1], [1, 1], [2, 2]]

  #groups per thread block
  g16 = 16
  g32 = 32
  g64 = 64

  #output shape per thread block
  npq_1x4x4 = [1, 4, 4]
  npq_1x8x8 = [1, 8, 8]
  npq_1x10x10 = [1, 10, 10]

  tile_descriptions = []
  for math_inst in math_instructions:
    for stride, dilation in product(strides, dilations):
      tile_descriptions.extend([
        # filter3x3               ThreadBlock_output, filter, stage, warp 
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g32], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g64], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g16], filter_3x3, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x10x10+[g64], filter_3x3, 2, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g32], filter_3x3, 4, stride, dilation, [4, 1, 1],  math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g64], filter_3x3, 4,  stride, dilation,[4, 1, 1], math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g16], filter_3x3, 4, stride, dilation, [4, 1, 1],  math_inst, min_cc, max_cc),

        # filter5x5               ThreadBlock_output, filter, stage, warp 
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g32], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g64], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x8x8+[g16], filter_5x5, 3, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x10x10+[g64], filter_5x5, 2, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),

        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g32], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g64], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc),
        Direct2dConvFixedStrideDilationTileDescription(npq_1x4x4+[g16], filter_5x5, 4, stride, dilation,[4, 1, 1],math_inst, min_cc, max_cc)
      ])

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateDepthwiseConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM60(manifest, cuda_version):
  GenerateSM60_Simt(manifest, cuda_version)
  GenerateSM60_Simt_DepthwiseConv2d(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM61_Simt(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 4],                                      \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 61
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 32], 2, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 32], 2, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints)
#

#
def GenerateSM61(manifest, cuda_version):
  GenerateSM61_Simt(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM70_TensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 75

  alignment_constraints = [8, 4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)
    
      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)

#
def GenerateSM70_PlanarComplexTensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [8, 8, 4],                                      \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 75

  alignment_constraints = [8, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)
    

#
def GenerateSM70_WmmaTensorOp_161616(manifest, cuda_version):

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
  ]

  min_cc = 70
  max_cc = 1024

  alignment_constraints = [8,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

#
##################################################################################################
#

def GenerateSM70(manifest, cuda_version):
  GenerateSM70_TensorOp_884(manifest, cuda_version)
  GenerateSM70_PlanarComplexTensorOp_884(manifest, cuda_version)

  # To limit build size, WMMA GEMMs are disabled for now.
  #
  #GenerateSM70_WmmaTensorOp_161616(manifest, cuda_version)

###################################################################################################
###################################################################################################

#
def GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst):

  min_cc = 75
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128,  64, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 64], 2, [2, 2, 2], math_inst, min_cc, max_cc),
  ]

  data_type = [
    math_inst.element_a,
    math_inst.element_b,
    math_inst.element_accumulator,
    math_inst.element_accumulator,
  ]

  conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

  CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [4, 8])
  CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [1, 2, 4])

  # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
  if math_inst.element_a != math_inst.element_accumulator:

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]

    CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [4, 8])
    CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [1, 2, 4])

#
def GenerateSM75_TensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [8, 4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [1, 2, 2], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)

    # Separate generator for 'few channels' specializations
    GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst)

#

#
def GenerateSM75_PlanarComplexTensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [8, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)

#
def GenerateSM75_TensorOp_8816_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [16,]
  alignment_constraints_small_channels = [16, 8, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),      
      TileDescription([ 64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.s32,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = []

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)
    
      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      operations += CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        if op.tile_description.threadblock_shape[1] >= 128:
          op.C.alignment = 16
        else:
          op.C.alignment = 8

#

#
def GenerateSM75_TensorOp_8816_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved32, LayoutType.RowMajorInterleaved32, LayoutType.ColumnMajorInterleaved32),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 16],                                     \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      DataType.f32,
    ]
    
    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

    conv_layout = (LayoutType.TensorNC32HW32, LayoutType.TensorC32RSK32, LayoutType.TensorNC32HW32)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    for op in operations:
      op.C.alignment = 8
#

#
def GenerateSM75_TensorOp_8832_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 1024
  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.s32,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)
      
    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = []

      operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)
    
      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        if op.tile_description.threadblock_shape[1] >= 128:
          op.C.alignment = 16
        elif op.tile_description.threadblock_shape[1] == 64:
          op.C.alignment = 8
        else:
          op.C.alignment = 8

#

#
def GenerateSM75_TensorOp_8832_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved64, LayoutType.RowMajorInterleaved64, LayoutType.ColumnMajorInterleaved64),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [8, 8, 32],                                     \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 75
  max_cc = 1024
  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

      conv_layout = (LayoutType.TensorNC64HW64, LayoutType.TensorC64RSK64, LayoutType.TensorNC64HW64)
  
      operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

      for op in operations:
        op.C.alignment = 16 
#

#
def GenerateSM75_TensorOp_88128(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [8, 8, 128],                                   \
      DataType.b1, DataType.b1, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.xor_popc),
  ]

  min_cc = 75 
  max_cc = 1024
  alignment_constraints = [128,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 512], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 512], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 512], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 512], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 512], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.b1, DataType.b1, DataType.s32, DataType.s32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

#

#
def GenerateSM75_WmmaTensorOp_161616(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 10, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 16, 16],                                   \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.WmmaTensorOp,                       \
      MathOperation.multiply_add),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      DataType.f32,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        DataType.f32,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)
#

#
def GenerateSM75_Simt_complex(manifest, cuda_version):
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 75
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc)
    ]
    data_type = [
      DataType.cf32,
      DataType.cf32,
      DataType.cf32,
      DataType.cf32
    ]

    complex_transforms = [
      (ComplexTransform.none, ComplexTransform.none),
      (ComplexTransform.conj, ComplexTransform.none),
      (ComplexTransform.none, ComplexTransform.conj),
      (ComplexTransform.conj, ComplexTransform.conj)
    ]

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

def GenerateSM75(manifest, cuda_version):
  GenerateSM75_TensorOp_1688(manifest, cuda_version)
  GenerateSM75_PlanarComplexTensorOp_1688(manifest, cuda_version)
  GenerateSM75_TensorOp_8816_TN(manifest, cuda_version)
  GenerateSM75_TensorOp_8816_Interleaved(manifest, cuda_version)
  GenerateSM75_TensorOp_8832_TN(manifest, cuda_version)
  GenerateSM75_TensorOp_8832_Interleaved(manifest, cuda_version)
  GenerateSM75_TensorOp_88128(manifest, cuda_version)
  #GenerateSM75_WmmaTensorOp_161616(manifest, cuda_version)
  GenerateSM75_Simt_complex(manifest, cuda_version)


###################################################################################################
###################################################################################################

#
def GenerateSM80_TensorOp_16816(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8, 4, 2]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  3, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
    CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type, [4, 8])
    CreateConv3dOperator(manifest, LayoutType.TensorNDHWC, tile_descriptions, data_type, 8)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)
      CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, [4, 8])
      CreateConv3dOperator(manifest, LayoutType.TensorNDHWC, tile_descriptions, data_type_mixed, 8)
#

#
def GenerateSM80_SparseTensorOp_16832(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

#

#
def GenerateSM80_PlanarComplexTensorOp_16816(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.bf16, DataType.bf16, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8, ]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([ 64, 128, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, complex_transforms)

#
def GenerateSM80_TensorOp_16832_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  smem_usage = 164

  alignment_constraints = [16,]
  alignment_constraints_small_channels = [16, 8, 4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256,  64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64,  64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256,  64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128,  64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128,  64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64,  64], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]
  
    data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, DataType.s32]
    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)
 
    operations = []

    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)
    
    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    operations += CreateConv2dFixedChannelsOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)

    operations += CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints_small_channels, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)
 
    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        op.C.alignment = 16
      else:
        op.C.alignment = 8

#

#
def GenerateSM80_SparseTensorOp_16864_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
  ]

  math_inst =                                         \
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [16,]

  tile_descriptions = [
    TileDescription([128,  64, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 128],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 128],  3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 128],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.s8, DataType.s8, DataType.s32, DataType.s32]
  data_type_mixed = [DataType.s8, DataType.s8, DataType.s8, DataType.f32]

  CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

  operations = []

  operations += CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

  for op in operations:
    if op.tile_description.threadblock_shape[1] >= 128:
      op.C.alignment = 16
    else:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16832_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajorInterleaved32, LayoutType.RowMajorInterleaved32, LayoutType.ColumnMajorInterleaved32),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.s8, DataType.s8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 32],                                    \
      DataType.u8, DataType.u8, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [16,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 64],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 64],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64], 10, [2, 2, 1], math_inst, min_cc, max_cc),
    ]
  
    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]
  
    operations = CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)
 
    conv_layout = (LayoutType.TensorNC32HW32, LayoutType.TensorC32RSK32, LayoutType.TensorNC32HW32)

    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)
 
    for op in operations:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16864_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 128], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 256],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 256],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 256],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 256],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 256],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]
  
    data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, DataType.s32]
    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]
  
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)
  
    operations = []
  
    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)
       
    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombination)
    
    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)
  
    for op in operations:
      if op.tile_description.threadblock_shape[1] >= 128:
        op.C.alignment = 16
      elif op.tile_description.threadblock_shape[1] == 64:
        op.C.alignment = 8
      else:
        op.C.alignment = 8
#

#
def GenerateSM80_SparseTensorOp_168128_TN(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
  ]

  math_inst =                                         \
    MathInstruction(                                  \
      [16, 8, 128],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate)

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  tile_descriptions = [
    TileDescription([ 64,  64, 256],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256,  64, 256],  3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 128, 256],  3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 256, 256],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 256, 256],  4, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 256],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 256],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 128, 512],  3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128,  64, 512],  4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64, 128, 512],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([ 64,  64, 512],  3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.s4, DataType.s4, DataType.s32, DataType.s32]
  data_type_mixed = [DataType.s4, DataType.s4, DataType.s4, DataType.f32]

  CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, None, EpilogueFunctor.LinearCombination)

  operations = []

  operations += CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
    data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)

  for op in operations:
    if op.tile_description.threadblock_shape[1] > 128:
      op.C.alignment = 16
    else:
      op.C.alignment = 8
#

#
def GenerateSM80_TensorOp_16864_Interleaved(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
      (LayoutType.ColumnMajorInterleaved64, LayoutType.RowMajorInterleaved64, LayoutType.ColumnMajorInterleaved64),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.s4, DataType.s4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
    MathInstruction(                                  \
      [16, 8, 64],                                    \
      DataType.u4, DataType.u4, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add_saturate),
  ]

  min_cc = 80
  max_cc = 1024
  alignment_constraints = [32,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 128],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 128],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 128],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 128],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 128],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 128],  6, [2, 2, 1], math_inst, min_cc, max_cc),
    ]
  
    data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, DataType.f32]
    
    operations = []
  
    operations += CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints, None, EpilogueFunctor.LinearCombinationClamp)
 
    conv_layout = (LayoutType.TensorNC64HW64, LayoutType.TensorC64RSK64, LayoutType.TensorNC64HW64)
  
    operations += CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
      data_type_mixed, alignment_constraints, [ConvKind.Fprop], EpilogueFunctor.LinearCombinationClamp)
 
    for op in operations:
      op.C.alignment = 16 
#

#
def GenerateSM80_TensorOp_168256(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 256],                                   \
      DataType.b1, DataType.b1, DataType.s32,         \
      OpcodeClass.TensorOp,                           \
      MathOperation.xor_popc),
  ]

  min_cc = 80
  max_cc = { 
    MathOperation.xor_popc: 1024
  }

  alignment_constraints = [128,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128,  512],  3, [4, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 256,  512],  3, [2, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256,  64,  512],  4, [4, 1, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 256,  512],  4, [1, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 128,  512],  5, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128,  64,  512],  6, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 128,  512],  6, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64,  64,  512], 10, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256, 128, 1024],  3, [4, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 256, 1024],  3, [2, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([256,  64, 1024],  4, [4, 1, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 256, 1024],  4, [1, 4, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128, 128, 1024],  4, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([128,  64, 1024],  3, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64, 128, 1024],  3, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
      TileDescription([ 64,  64, 1024],  5, [2, 2, 1], math_inst, min_cc, max_cc[math_inst.math_operation]),
    ]

    data_type = [DataType.b1, DataType.b1, DataType.s32, DataType.s32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

#

#
def GenerateSM80_TensorOp_1688(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),            
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64,  128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    data_type_mixed = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_a,
      math_inst.element_accumulator,
    ]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type_mixed, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.f16, DataType.f16, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f16),
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.bf16, DataType.bf16, DataType.f32,       \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_bf16),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),            
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 16],  4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

def GenerateSM80_TensorOp_1688_fast_fp32_math_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(                            \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32)

  min_cc = 80
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [
    DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
  ]

  alignment_constraints = [1,]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)


#
def GenerateSM80_SparseTensorOp_16816_fast_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 1):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 16],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 64],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 64],  3, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 64],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 64],  3, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateSparseGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(                  \
    [16, 8, 8],                                 \
    DataType.tf32, DataType.tf32, DataType.f32,   \
    OpcodeClass.TensorOp,                       \
    MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [
    DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
  ]

  alignment_constraints = [1,]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_1688_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [ 
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1, 2, 4]  # Alignment only applies to A in SYRK

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32]

    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_1688_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [ 
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    # SYRK
    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)

    # HERK
    CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_1688_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_instructions = [ 
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1, 2, 4] 

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
      data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_1688_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_instructions = [ 
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    complex_transforms = [
      ComplexTransform.none, ComplexTransform.conj,
    ]

    CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
      data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_1688_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  # A and B have same layouts
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [ 
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [
    1, 2, 4
  ] 

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 16],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 16],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 16],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 16],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 16],  5, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 32],  3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32],  3, [2, 4, 1], math_inst, min_cc, max_cc),
      #TileDescription([256,  64, 32],  4, [4, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 256, 32],  4, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32],  4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([128,  64, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64, 128, 32],  3, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([ 64,  64, 32],  5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_1688_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_instructions = [ 
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,         \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex),
      MathInstruction(                                    \
      [16, 8, 8],                                         \
      DataType.f32, DataType.f32, DataType.f32,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_fast_f32),
  ]

  min_cc = 80
  max_cc = 1024

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 64, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 128, 16], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      #TileDescription([64, 32, 16], 4, [2, 1, 1], math_inst, min_cc, max_cc),
      #TileDescription([32, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      DataType.cf32, DataType.cf32, DataType.cf32, DataType.cf32
    ]

    alignment_constraints = [1,]

    # SYMM
    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.symmetric)

    # HEMM
    CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
      data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 32, 16], 3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 256, 16], 3, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_884_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64,  8 ], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 8 ], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  8 ], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  8 ], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  8 ], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64,  16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  16], 3, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)

#
def GenerateSM80_TensorOp_884_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_884_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper, 
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64]

  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_884_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)

#

#
def GenerateSM80_TensorOp_884_rank_k_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM80_TensorOp_884_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#


#
def GenerateSM80_TensorOp_884_trmm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM80_TensorOp_884_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper, 
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM80_TensorOp_884_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM80_TensorOp_884_symm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [8, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

###################################################################################################

#
def GenerateSM80_Simt_f32(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 5, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([256, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 8], 4, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 5, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#


#
def GenerateSM80_Simt_f64(manifest, cuda_version):
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f64, DataType.f64, DataType.f64,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, 128, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  32, 8], 5, [2, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 32, 128, 8], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)
#


##################################################################################################
#
def GenerateSM80_Simt_complex(manifest, cuda_version):
  math_instructions = [
    MathInstruction(                                  \
      [1, 1, 1],                                      \
      DataType.f32, DataType.f32, DataType.f32,       \
      OpcodeClass.Simt,                               \
      MathOperation.multiply_add_complex),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [1,]

  data_type = [
    DataType.cf32,
    DataType.cf32,
    DataType.cf32,
    DataType.cf32
  ]
    
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  for math_inst in math_instructions:

    tile_descriptions = [
      TileDescription([128, 128, 8], 5, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 8], 4, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 16],  6, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, alignment_constraints, complex_transforms)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)
#

###################################################################################################

#
def GenerateSM80(manifest, cuda_version):
  GenerateSM80_TensorOp_16816(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16832(manifest, cuda_version)
  GenerateSM80_PlanarComplexTensorOp_16816(manifest, cuda_version)
  GenerateSM80_TensorOp_1688(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16816_fast_math(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_complex(manifest, cuda_version)
  # 3xTF32 
  GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_fast_fp32_math_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_rank_k(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_rank_k_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_trmm(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_trmm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_symm(manifest, cuda_version)
  GenerateSM80_TensorOp_1688_symm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884(manifest, cuda_version)
  GenerateSM80_TensorOp_884_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_rank_k_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_trmm_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm_complex(manifest, cuda_version)
  GenerateSM80_TensorOp_884_symm_complex_gaussian(manifest, cuda_version)
  GenerateSM80_TensorOp_16832_TN(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_16864_TN(manifest, cuda_version)
  GenerateSM80_TensorOp_16832_Interleaved(manifest, cuda_version)
  GenerateSM80_TensorOp_16864_TN(manifest, cuda_version)
  GenerateSM80_SparseTensorOp_168128_TN(manifest, cuda_version)
  GenerateSM80_TensorOp_16864_Interleaved(manifest, cuda_version)
  GenerateSM80_TensorOp_168256(manifest, cuda_version)
  GenerateSM80_Simt_f32(manifest, cuda_version)
  GenerateSM80_Simt_f64(manifest, cuda_version)
  GenerateSM80_Simt_complex(manifest, cuda_version)

###################################################################################################

#
def GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = [
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.bf16, DataType.bf16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
      TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
      TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
      TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type)

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed)


#
def GenerateSM90_TensorOp_tf32_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments
  layouts_tf32 = [
    [[LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 1], [LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    4], [LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1]],
  ]

  math_inst = MathInstruction(
      [64, 128, 8],
      DataType.tf32, DataType.tf32, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 90

  tile_descriptions = [
    TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
    TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
    TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
      0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
  ]

  data_type_tf32 = [
    math_inst.element_a,
    math_inst.element_b,
    math_inst.element_accumulator,
    math_inst.element_accumulator,
  ]

  CreateGemmUniversal3xOperator(manifest, layouts_tf32, tile_descriptions, data_type_tf32)

  # F32 kernel, TN only supported for now
  layouts_f32 = [layouts_tf32[2]]

  data_type_f32 = [
    DataType.f32,
    DataType.f32,
    math_inst.element_accumulator,
    DataType.f32,
  ]

  CreateGemmUniversal3xOperator(manifest, layouts_f32, tile_descriptions, data_type_f32)


def GenerateSM90_TensorOp_int8_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return
  
  # layouts for ABC and their alignments
  layouts = [
    [[LayoutType.RowMajor, 16], [LayoutType.ColumnMajor, 16], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = [
    MathInstruction(
      [64, 128, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 32],
      DataType.u8, DataType.u8, DataType.s32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
      TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
      TileDescription([128, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
      TileDescription([ 64, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type)

#
def GenerateSM90_TensorOp_1684(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst = MathInstruction(
      [16, 8, 4],
      DataType.f64, DataType.f64, DataType.f64,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([256, 32, 16], 3, [4, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 256, 16], 3, [1, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateGemmOperator(manifest, layouts, tile_descriptions,
    data_type, alignment_constraints)

#

#
def GenerateSM90_TensorOp_1684_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64,  8 ], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 8 ], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  8 ], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  8 ], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  8 ], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  8 ], 4, [2, 1, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64,  16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  128, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64,  32,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  64,  16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  32,  16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16,  32,  16], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32,  16,  16], 3, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    (ComplexTransform.none, ComplexTransform.none),
    (ComplexTransform.conj, ComplexTransform.none),
    (ComplexTransform.none, ComplexTransform.conj),
    (ComplexTransform.conj, ComplexTransform.conj)
  ]

  CreateGemmOperator(manifest, layouts, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_rank_k(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper, 
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64]

  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM90_TensorOp_1684_rank_k_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)

#

#
def GenerateSM90_TensorOp_1684_rank_k_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYRK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HERK computation
  CreateRankKOperator(manifest, layouts, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM90_TensorOp_1684_trmm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints)
#

#
def GenerateSM90_TensorOp_1684_trmm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#


#
def GenerateSM90_TensorOp_1684_trmm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  diag_types = [
    DiagType.NonUnit, DiagType.Unit,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [
    ComplexTransform.none, ComplexTransform.conj,
  ]

  CreateTrmmOperator(manifest, layouts, side_modes, fill_modes, diag_types, tile_descriptions, \
    data_type, alignment_constraints, complex_transforms)
#

#
def GenerateSM90_TensorOp_1684_symm(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper, 
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 32, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([16, 32, 16], 5, [1, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 16, 16], 5, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.f64, DataType.f64, DataType.f64, DataType.f64]

  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)
#

#
def GenerateSM90_TensorOp_1684_symm_complex(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([128, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 128, 8], 3, [2, 4, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 64, 8], 3, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

#
def GenerateSM90_TensorOp_1684_symm_complex_gaussian(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 8):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  ]

  side_modes = [
    SideMode.Left, SideMode.Right,
  ]

  fill_modes = [
    FillMode.Lower, FillMode.Upper,
  ]

  math_inst =                                             \
    MathInstruction(                                      \
      [16, 8, 4],                                          \
      DataType.f64, DataType.f64, DataType.f64,           \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add_complex_gaussian)

  min_cc = 90
  max_cc = 1024

  alignment_constraints = [1,]

  tile_descriptions = [
    TileDescription([64, 64, 8], 3, [4, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([64, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    TileDescription([32, 64, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 32, 8], 4, [2, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([16, 32, 8], 4, [1, 2, 1], math_inst, min_cc, max_cc),
    #TileDescription([32, 16, 8], 4, [2, 1, 1], math_inst, min_cc, max_cc),
  ]

  data_type = [DataType.cf64, DataType.cf64, DataType.cf64, DataType.cf64]

  complex_transforms = [ComplexTransform.none,]

  # SYMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.symmetric)

  # HEMM computation
  CreateSymmOperator(manifest, layouts, side_modes, fill_modes, tile_descriptions, \
    data_type, alignment_constraints, BlasMode.hermitian)
#

###################################################################################################

#
def GenerateSM90(manifest, cuda_version):
  GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_int8_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_tf32_WGMMA_gemm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_rank_k_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_trmm_complex_gaussian(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm_complex(manifest, cuda_version)
  GenerateSM90_TensorOp_1684_symm_complex_gaussian(manifest, cuda_version)

###################################################################################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--operations", default="all", help="Specifies the operation to generate (gemm, all)")
  parser.add_argument("--build-dir", default=".", required=False, help="CUTLASS top-level build directory")
  parser.add_argument("--curr-build-dir", default=".", help="CUTLASS current build directory. cmake files will be emitted in this directory")
  parser.add_argument("--generator-target", default='library', help="Target of CUTLASS Library Generator.")
  parser.add_argument("--architectures", default='53;60;61;70;75;80', help="Target compute architectures")
  parser.add_argument("--kernels", default='', help='Comma delimited list to filter kernels by name.')
  parser.add_argument("--ignore-kernels", default='', help='Comma delimited list of kernels to exclude from build.')
  parser.add_argument("--filter-by-cc", default='True', type=str, help='If enabled, kernels whose comupte capability range is not satisfied by the build target are excluded.')
  parser.add_argument("--cuda-version", default="11.0.0", help="Semantic version string of CUDA Toolkit")
  parser.add_argument('--kernel-filter-file',   type=str, default=None, required=False, help='Full path of filter file')
  parser.add_argument('--selected-kernel-list',   type=str, default=None, required=False,
                        help='Specify the output log file containing all enabled kernels in this build')
  parser.add_argument("--interface-dir", default=None, required=False, help="Interface header to kernels")
  parser.add_argument("--disable-full-archs-compilation", action="store_true", required=False, help="Disable compilation for every archs in --architectures")

  def numeric_log_level(log_level: str) -> int:
    """
    Converts the string identifier of the log level into the numeric identifier used
    in setting the log level

    :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
    :type x: str

    :return: numeric representation of log level
    :rtype: int
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
      raise ValueError(f'Invalid log level: {log_level}')
    return numeric_level

  parser.add_argument("--log-level", default='info', type=numeric_log_level, required=False,
                      help='Logging level to be used by the generator script')

  args = parser.parse_args()

  # Set the logging level based on the user-provided `--log-level` command-line option
  logging.basicConfig(level=args.log_level)

  manifest = Manifest(args)

  GenerateSM50(manifest, args.cuda_version)
  GenerateSM60(manifest, args.cuda_version)
  GenerateSM61(manifest, args.cuda_version)
  GenerateSM70(manifest, args.cuda_version)
  GenerateSM75(manifest, args.cuda_version)
  GenerateSM80(manifest, args.cuda_version)
  GenerateSM90(manifest, args.cuda_version)

  if 'library' in args.generator_target.split(','):
    manifest.emit(GeneratorTarget.Library)

  if args.selected_kernel_list is not None:
    if len(manifest.selected_kernels) > 0:
      with open(args.selected_kernel_list, 'w') as file_writer:
        for line in manifest.selected_kernels:
          file_writer.write("%s\n" % line)
#
###################################################################################################
