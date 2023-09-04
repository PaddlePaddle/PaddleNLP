################################################################################
#
# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################
from typeguard import typechecked
from cuda import cuda
from typing import Union
import numpy as np

from typeguard import typechecked

from pycutlass import *


# @typechecked
class Conv2dArguments(ArgumentBase):
    """
    Argument wrapper for Conv2d. It encodes problem information and 
    user-provide tensors into the kernel's argument.

    :param operation: the Conv2d operation to take the argument
    :type operation: :class:`pycutlass.Conv2dOperation`

    :param problem_size: the Conv2d problem size
    :type problem_size: :class:`cutlass.conv.Conv2dProblemSize`

    :param A: tensor A
    :type A: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param B: tensor B
    :type B: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param C: tensor C
    :type C: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param D: tensor D
    :type D: cuda.CUdeviceptr | numpy.ndarray | torch.Tensor | cupy.ndarray

    :param split_k_mode: conv2d split K mode, defaults to 
    cutlass.conv.SplitKMode.Serial
    :type split_k_mode: cutlass.conv.SplitKMode, optional

    :param output_op: output operator, optional
    :type output_op: :class:`pycutlass.LinearCombinationFunctorArguments`

    """

    def __init__(self, operation: 'Conv2dOperation',
                 problem_size: 'cutlass.conv.Conv2dProblemSize',
                 A: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor]',
                 B: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor]',
                 C: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor]',
                 D: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor]',
                 split_k_mode: 'cutlass.conv.SplitKMode'
                    = cutlass.conv.SplitKMode.Serial, **kwargs) -> None:

        self.operation = operation
        #: convolution kind
        self.conv_kind: cutlass.conv.Operator = operation.conv_kind
        self.layout_A: cutlass.layout = operation.A.layout
        self.layout_B: cutlass.layout = operation.B.layout
        self.layout_C: cutlass.layout = operation.C.layout

        self.element_A = operation.A.element
        self.element_B = operation.B.element
        self.element_C = operation.C.element

        if self.layout_C == cutlass.TensorNC32HW32:
            B = self.reorder_tensor_B(B, problem_size)

        super().__init__(A, B, C, D, **kwargs)
        # preprocessing output ops
        
        if 'output_op' in kwargs.keys() and \
            split_k_mode != cutlass.conv.SplitKMode.Parallel:
            self.output_op = kwargs['output_op']
        else:
            self.output_op = self.operation.epilogue_type(1.0, 0.0)

        if "split_k_slices" in kwargs.keys():
            self.split_k_mode = split_k_mode
            self.split_k_slices = kwargs["split_k_slices"]
        else:
            self.split_k_mode = cutlass.conv.SplitKMode.Serial
            self.split_k_slices = 1

        #: problem_size
        self.problem_size: cutlass.conv.Conv2dProblemSize = problem_size
        self.problem_size.split_k_slices = self.split_k_slices

        if hasattr(self, "tensor_c_numel"):
            c_coord = cutlass.conv.implicit_gemm_tensor_c_extent(
                self.conv_kind, problem_size)
            if (self.tensor_c_numel == c_coord.at(3) and 
                self.tensor_c_numel < c_coord.size()):
                self.bias = True

        #
        # initialize the argument
        #
        self.initialize()

    # @typechecked
    def reorder_tensor_B(self, tensor_B: 'np.ndarray', 
            problem_size: 'cutlass.conv.Conv2dProblemSize'):
        """
        Reorder tensor_B for interleaved layout

        :param tensor_B: input tensor B
        :type tensor_B: numpy.ndarray
        :param problem_size: Conv2d problem size
        :type problem_size: :class:`cutlass.conv.Conv2dProblemSize`

        :return: reordered tensor B
        :rtype: numpy.ndarray
        """
        reordered_tensor_B = np.empty_like(tensor_B)
        tensor_ref_B = self.get_tensor_ref(
            tensor_B, self.element_B, self.layout_B, problem_size, "b")
        reordered_tensor_ref_B = self.get_tensor_ref(
            reordered_tensor_B, self.element_B, 
            self.layout_B, problem_size, "b")
        cutlass.conv.host.reorder_convK(
            reordered_tensor_ref_B, tensor_ref_B, self.conv_kind, problem_size)

        return reordered_tensor_B

    def get_tensor_ref(
        self, tensor, dtype, tensor_layout, problem_size, operand):
        if operand == "a":
            tensor_coord = cutlass.conv.implicit_gemm_tensor_a_extent(
                self.conv_kind, problem_size)
        elif operand == "b":
            tensor_coord = cutlass.conv.implicit_gemm_tensor_b_extent(
                self.conv_kind, problem_size)
        elif operand in ["c", "d"]:
            tensor_coord = cutlass.conv.implicit_gemm_tensor_c_extent(
                self.conv_kind, problem_size)
        else:
            raise ValueError("unknown operand: " + operand)
        # Zero stride trick
        if operand == "c" and self.bias:
            tensor_coord = cutlass.Tensor4DCoord(0, 0, 0, 0)

        layout = tensor_layout.packed(tensor_coord)

        return TensorRef(tensor, dtype, layout).tensor_ref

    def get_arguments(self, semaphore):
        ref_A = TensorRef_(self.get_tensor_ref(
            self.ptr_A, self.element_A, self.layout_A, self.problem_size, "a"))
        ref_B = TensorRef_(self.get_tensor_ref(
            self.ptr_B, self.element_B, self.layout_B, self.problem_size, "b"))
        ref_C = TensorRef_(self.get_tensor_ref(
            self.ptr_C, self.element_C, self.layout_C, self.problem_size, "c"))
        ref_D = TensorRef_(self.get_tensor_ref(
            self.ptr_D, self.element_C, self.layout_C, self.problem_size, "d"))

        self.c_arguments = self.operation.argument_type(
            Conv2DProblemSize(self.problem_size),
            ref_A, ref_B, ref_C, ref_D, self.output_op, self.split_k_mode
        )

        self.semaphore = semaphore

    def initialize(self):
        """
        Initialize the kernel arguments handling following stuffs
        1. get kernel launch configuration including grid, cta size, 
           and dynamic shared memory capacity
        2. allocate and initialize device workspace
        3. get kernel params as bytearray for NVRTC input
        """
        # get launch configuration
        self.launch_config = self.operation.rt_module.plan(self)

        # allocate and initialize device workspace
        device_workspace_size = \
            self.operation.rt_module.get_device_workspace_size(self)

        if device_workspace_size > 0:
            self.workspace_buffer = device_mem_alloc(device_workspace_size)
            workspace_ptr = self.workspace_buffer.ptr
            err, = cuda.cuMemsetD32(
                workspace_ptr, 0, device_workspace_size // 4)
        else:
            workspace_ptr = None

        # get kernel params as bytearray
        semaphore = 0
        if workspace_ptr is not None and \
            self.split_k_mode == cutlass.conv.SplitKMode.Parallel:
            self.ptr_D = workspace_ptr
        elif workspace_ptr is not None and \
            self.split_k_mode == cutlass.conv.SplitKMode.Serial:
            semaphore = workspace_ptr

        self.get_arguments(semaphore)

        params_ = self.operation.rt_module.get_args(ctypes.byref(
            self.c_arguments), ctypes.c_void_p(int(self.semaphore)))
        self.host_workspace = bytearray(params_.contents)
        self.device_workspace = None

    def sync(self):
        """
        Synchronize the arguments. If the input tensor is in host, 
        copy it from device to host.
        """
        return super().sync()


# @typechecked
class Conv2dRT(ExecutableOperation):
    """
    Conv2dRT manages the CUTLASS runtime components
    """
    KernelTemplate = r'''
extern "C"
__global__ void
${operation_name}(${operation_name}${operation_suffix}::Params params) {

  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  ${operation_name}${operation_suffix}::SharedStorage *shared_storage =
      reinterpret_cast<${operation_name}${operation_suffix}::SharedStorage *>(SharedStorageBase);

  ${operation_name}${operation_suffix} op;

  op(params, *shared_storage);
}
    '''

    HostTemplate = r'''
extern "C" {
  // Get the size of params in bytes
  int ${operation_name}_get_param_size(){
    return sizeof(${operation_name}${operation_suffix}::Params);
  }

  // Get the size of dynamic shared memory in bytes
  int ${operation_name}_shared_memory_size() {
    return int(sizeof(${operation_name}${operation_suffix}::SharedStorage));
  }

  // Get the params as byte array
  char* ${operation_name}_get_params(${operation_name}${operation_suffix}::Arguments* arguments, int *semaphore=nullptr){
    typename ${operation_name}${operation_suffix}::Params* params;
    params = new ${operation_name}${operation_suffix}::Params(*arguments, semaphore);

    char *bytes = ((char*)(params));
    char *output = new char[sizeof(${operation_name}${operation_suffix}::Params)];
    for (unsigned int i = 0; i < sizeof(${operation_name}${operation_suffix}::Params); i ++)
        output[i] = bytes[i];

    return output;
  }
}

    '''

    def __init__(self, operation: 'Conv2dOperation'):
        super().__init__(operation)
        self.argument_type, self.epilogue_type = get_conv2d_arguments(operation.epilogue_functor)
        self.argtype = [ctypes.POINTER(self.argument_type), ctypes.c_void_p]
        self.conv_kind = operation.conv_kind

        self.operation: Conv2dOperation = operation

        self.emitter = EmitConv2dInstance('_type')

        self.threads: int = operation.tile_description.num_threads

        self.swizzle_functor = operation.swizzling_functor

    def emit(self):
        return self.emitter.emit(self.operation)

    # @typechecked
    def get_device_workspace_size(self, arguments: Conv2dArguments):
        workspace_bytes = 0

        launch_config = arguments.launch_config

        self.conv_kind = self.operation.conv_kind

        if arguments.split_k_mode == cutlass.conv.SplitKMode.Parallel:
            problem_size = arguments.problem_size
            workspace_bytes = DataTypeSize[self.operation.C.element] \
            * launch_config.grid[2] * cutlass.conv.implicit_gemm_tensor_c_size(
                self.conv_kind, problem_size
            ) // 8
        elif arguments.split_k_mode == cutlass.conv.SplitKMode.Serial and \
            arguments.split_k_slices > 1:
            workspace_bytes = launch_config.grid[0] * launch_config.grid[1] * 4

        return workspace_bytes

    # @typechecked
    def plan(self, arguments: Conv2dArguments):
        tile_size = cutlass.gemm.GemmCoord(
            self.operation.tile_description.threadblock_shape[0],
            self.operation.tile_description.threadblock_shape[1],
            self.operation.tile_description.threadblock_shape[2]
        )

        grid = self.swizzle_functor.get_grid_shape(
            self.swizzle_functor.get_tiled_shape(
                self.conv_kind, arguments.problem_size, 
                tile_size, arguments.split_k_slices
            )
        )
        return LaunchConfiguration(
            [grid.x, grid.y, grid.z], [self.threads, 1, 1], 
            self.shared_memory_capacity)

    def initialize(self):
        err, = cuda.cuFuncSetAttribute(
            self.kernel,
            attrib=cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            value=self.shared_memory_capacity)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))

#


class Conv2dOperation:
    """
    CUTLASS Conv2d operation description.

    :param conv_kind: convolution operator
    :type conv_kind: :class:`cutlass.conv.Operator`

    :param iterator_algorithm: Selects among several implementation 
    variants trading off performance with simplicity
    :type iterator_algorithm: :class:`cutlass.conv.IteratorAlgorithm`

    :param arch: GPU compute capability (sm_xx)
    :type arch: int

    :param tile_description: tile description
    :type tile_description: :class:`pycutlass.TileDescription`

    :param A: tensor A description
    :type A: :class:`pycutlass.TensorDescription`

    :param B: tensor B description
    :type B: :class:`pycutlass.TensorDescription`

    :param C: tensor C description
    :type C: :class:`pycutlass.TensorDescription`

    :param D: tensor D description
    :type D: :class:`pycutlass.TensorDescription`

    :param element_epilogue: element type for computation in epilogue \
    :type element_epilogue: cutlass.int8 | cutlass.int32 | cutlass.float16 | \
    cutlass.bfloat16 | cutlass.float32 | cutlass.float64

    :param stride_support: distinguish among partial specializations that \
    accelerate certain problems where convolution stride is unit \
    :type stride_support: :class:`cutlass.conv.StrideSupport`

    :param epilogue_functor: convolution epilogue functor
    :type epilogue_functor: :class:`EpilogueFunctor`

    :param swizzling_functor: threadblock swizzling functor
    """
    #

    def __init__(self,
                 conv_kind: cutlass.conv.Operator,
                 iterator_algorithm: cutlass.conv.IteratorAlgorithm,
                 arch: int, tile_description: TileDescription,
                 A: TensorDescription, B: TensorDescription, C: TensorDescription,
                 stride_support, epilogue_functor,
                 swizzling_functor=cutlass.IdentitySwizzle1):

        self.operation_kind: OperationKind = OperationKind.Conv2d
        self.arch: int = arch
        self.tile_description: TileDescription = tile_description
        self.conv_kind = conv_kind
        self.A: TensorDescription = A
        self.B: TensorDescription = B
        self.C: TensorDescription = C
        self.epilogue_functor = epilogue_functor
        self.iterator_algorithm = iterator_algorithm
        self.stride_support = stride_support
        self.swizzling_functor = swizzling_functor()

        self.rt_module: Conv2dRT = Conv2dRT(self)
        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type

    def run(self, arguments: Conv2dArguments) -> cuda.CUresult:
        """
        Launch the cuda kernel with input arguments

        :param arguments: conv2d arguments
        :type arguments: :class:`pycutlass.Conv2dArguments`
        """

        # launch the kernel
        err = self.rt_module.run(
            arguments.host_workspace,
            arguments.device_workspace,
            arguments.launch_config)

        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('CUDA Error %s' % str(err))

        return err

    #
    # Get function name
    #

    def procedural_name(self):
        ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
        return self.configuration_name()
    #

    def configuration_name(self):
        ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''

        opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

        threadblock = "%dx%d_%dx%d" % (
            self.tile_description.threadblock_shape[0],
            self.tile_description.threadblock_shape[1],
            self.tile_description.threadblock_shape[2],
            self.tile_description.stages
        )

        if self.stride_support == StrideSupport.Unity:
            configuration_name = "cutlass_sm${arch}_${opcode_class}_${extended_name}_${threadblock}_${layout}_unity_stride_align${alignment}"
        else:
            configuration_name = "cutlass_sm${arch}_${opcode_class}_${extended_name}_${threadblock}_${layout}_align${alignment}"

        return SubstituteTemplate(
            configuration_name,
            {
                'arch': str(self.arch),
                'opcode_class': opcode_class_name,
                'extended_name': self.extended_name(),
                'threadblock': threadblock,
                'layout': self.layout_name(),
                'alignment': "%d" % self.A.alignment,
            }
        )

    #
    def extended_name(self):
        ''' Append data types if they differ from compute type. '''
        if self.C.element != self.tile_description.math_instruction.element_accumulator and \
                self.A.element != self.tile_description.math_instruction.element_accumulator:
            extended_name = "${element_c}_${core_name}_${element_a}"
        elif self.C.element == self.tile_description.math_instruction.element_accumulator and  \
                self.A.element != self.tile_description.math_instruction.element_accumulator:
            extended_name = "${core_name}_${element_a}"
        else:
            extended_name = "${core_name}"

        extended_name = SubstituteTemplate(extended_name, {
            'element_a': DataTypeNames[self.A.element],
            'element_c': DataTypeNames[self.C.element],
            'core_name': self.core_name()
        })

        return extended_name

    #
    def layout_name(self):
        return "%s" % (ShortLayoutTypeNames[self.A.layout])

    #
    def core_name(self):
        ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

        intermediate_type = ''

        if self.tile_description.math_instruction.opcode_class == cutlass.OpClass.TensorOp:
            inst_shape = "%dx%dx%d" % tuple(
                self.tile_description.math_instruction.instruction_shape)
            if self.tile_description.math_instruction.element_a != self.A.element and \
                    self.tile_description.math_instruction.element_a != self.accumulator_type():
                intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]
        else:
            inst_shape = ''

        return "%s%s%s%s_%s" % (ShortDataTypeNames[self.accumulator_type()],
                                inst_shape, intermediate_type, ConvKindNames[self.conv_kind], IteratorAlgorithmNames[self.iterator_algorithm])

    #
    def is_complex(self):
        complex_operators = [
            MathOperation.multiply_add_complex,
            MathOperation.multiply_add_complex_gaussian
        ]
        return self.tile_description.math_instruction.math_operation in complex_operators

    #
    def accumulator_type(self):
        accum = self.tile_description.math_instruction.element_accumulator

        if self.is_complex():
            return get_complex_from_real(accum)

        return accum


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

class EmitConv2dInstance:
    def __init__(self, operation_suffix=''):
        self.operation_suffix = operation_suffix
        self.includes = [
            "cutlass/cutlass.h",
            "cutlass/conv/kernel/default_conv2d_fprop.h",
            "cutlass/conv/kernel/default_conv2d_dgrad.h",
            "cutlass/conv/kernel/default_conv2d_wgrad.h"
        ]
        self.template = """
// Conv2d${conv_kind_name} ${iterator_algorithm_name} kernel instance "${operation_name}"
using ${operation_name}_base = 
typename cutlass::conv::kernel::DefaultConv2d${conv_kind_name}<
  ${element_a}, 
  ${layout_a},
  ${element_b}, 
  ${layout_b},
  ${element_c}, 
  ${layout_c},
  ${element_accumulator},
  ${opcode_class},
  ${arch},
  cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
  cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k} >,
  cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
  ${epilogue_functor},
  ${swizzling_functor}, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
  ${stages},
  ${math_operator},
  ${iterator_algorithm},
  ${stride_support},
  ${align_a},
  ${align_b}
>::Kernel;

struct ${operation_name}${operation_suffix}:
  public ${operation_name}_base { };

"""

    def emit(self, operation):

        warp_shape = [int(operation.tile_description.threadblock_shape[idx] /
                          operation.tile_description.warp_count[idx]) for idx in range(3)]

        epilogue_vector_length = int(min(
            operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

        values = {
            'operation_name': operation.procedural_name(),
            'operation_suffix': self.operation_suffix,
            'conv_kind': ConvKindTag[operation.conv_kind],
            'conv_kind_name': ConvKindNames[operation.conv_kind].capitalize(),
            'element_a': DataTypeTag[operation.A.element],
            'layout_a': LayoutTag[operation.A.layout],
            'element_b': DataTypeTag[operation.B.element],
            'layout_b': LayoutTag[operation.B.layout],
            'element_c': DataTypeTag[operation.C.element],
            'layout_c': LayoutTag[operation.C.layout],
            'element_accumulator': DataTypeTag[operation.accumulator_type()],
            'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
            'arch': "cutlass::arch::Sm%d" % operation.arch,
            'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
            'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
            'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
            'warp_shape_m': str(warp_shape[0]),
            'warp_shape_n': str(warp_shape[1]),
            'warp_shape_k': str(warp_shape[2]),
            'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
            'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
            'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
            'epilogue_vector_length': str(epilogue_vector_length),
            'epilogue_functor': operation.epilogue_functor.emit(),
            'swizzling_functor': operation.swizzling_functor.tag(),
            'stages': str(operation.tile_description.stages),
            'iterator_algorithm': IteratorAlgorithmTag[operation.iterator_algorithm],
            'iterator_algorithm_name': IteratorAlgorithmNames[operation.iterator_algorithm].capitalize(),
            'stride_support': StrideSupportTag[operation.stride_support],
            'math_operator': 'cutlass::arch::OpMultiplyAddComplex' if operation.is_complex() else
            MathOperationTag[operation.tile_description.math_instruction.math_operation],
            'align_a': str(operation.A.alignment),
            'align_b': str(operation.B.alignment),
        }

        return SubstituteTemplate(self.template, values)
