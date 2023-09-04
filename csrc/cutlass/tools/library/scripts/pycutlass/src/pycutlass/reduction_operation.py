################################################################################
#
# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pycutlass import *
from pycutlass.c_types import get_reduction_params
import cutlass
from cuda import cuda
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
import numpy as np
from typing import Union
from cuda import cudart


class ReductionOperation:
    pass


class ReductionArguments:
    """
    Arguments of reduction
    """

    def __init__(self, operation: ReductionOperation,
                 problem_size: 'list[int]', partitions: int,
                 workspace: cuda.CUdeviceptr,
                 destination: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor]',
                 source: 'Union[cuda.CUdeviceptr, np.ndarray, torch.Tensor]', **kwargs) -> None:

        # tensor_C can be interpreted as the bias with bias=True in keyword args
        if "bias" in kwargs.keys():
            self.bias = kwargs["bias"]
        else:
            # by default, tensor_C is not bias
            self.bias = False

        self.operation = operation
        #: pointer to the workspace
        self.ptr_workspace = workspace

        #: number of split-k partitions
        self.partitions = partitions

        if isinstance(destination, np.ndarray):
            self.host_D = destination
            self.destination_buffer = NumpyFrontend.argument(destination, True)
            self.source_buffer = NumpyFrontend.argument(source, False)
            self.ptr_destination = cuda.CUdeviceptr(
                self.destination_buffer.ptr)
            self.ptr_source = cuda.CUdeviceptr(self.source_buffer.ptr)
        elif torch_available and isinstance(destination, torch.Tensor):
            self.ptr_destination = TorchFrontend.argument(destination)
            self.ptr_source = TorchFrontend.argument(source)
        elif isinstance(destination, cuda.CUdeviceptr):
            self.ptr_destination = destination
            self.ptr_source = source
        else:
            raise TypeError("unknown Type")

        self.problem_size = MatrixCoord_(
            problem_size[0], problem_size[1]
        )

        self.partition_stride = problem_size[0] * \
            problem_size[1] * DataTypeSize[operation.C.element] // 8

        if "output_op" in kwargs.keys():
            self.output_op = kwargs['output_op']
        else:
            self.output_op = self.operation.epilogue_type(1.0, 0.0)

        # get arguments
        self.get_arguments()

    @staticmethod
    def get_tensor_ref(extent: 'tuple[int]', device_ptr: cuda.CUdeviceptr, layout: cutlass.layout):
        if layout == cutlass.RowMajor:
            return TensorRef2D_(int(device_ptr), extent[1])
        else:
            raise ValueError("unknonwn layout type")

    def get_arguments(self):
        ref_workspace = ReductionArguments.get_tensor_ref(
            extent=[self.problem_size.row, self.problem_size.column],
            device_ptr=self.ptr_workspace, layout=cutlass.RowMajor)
        if self.bias:
            ref_source = ReductionArguments.get_tensor_ref(
                extent=[0, 0],
                device_ptr=self.ptr_source, layout=cutlass.RowMajor)
        else:
            ref_source = ReductionArguments.get_tensor_ref(
                extent=[self.problem_size.row, self.problem_size.column],
                device_ptr=self.ptr_source, layout=cutlass.RowMajor)

        ref_destination = ReductionArguments.get_tensor_ref(
            extent=[self.problem_size.row, self.problem_size.column],
            device_ptr=self.ptr_destination, layout=cutlass.RowMajor)


        self.c_arguments = self.operation.argument_type(
            self.problem_size, self.partitions,
            self.partition_stride, ref_workspace,
            ref_destination, ref_source,
            self.output_op
        )

        params_ = self.operation.rt_module.get_args(
            ctypes.byref(self.c_arguments))
        self.host_workspace = bytearray(params_.contents)

    def sync(self):
        err, = cudart.cudaDeviceSynchronize()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA Error %s" % str(err))

        if hasattr(self, "host_D"):
            err, = cuda.cuMemcpyDtoH(
                self.host_D, self.ptr_destination, self.host_D.size * self.host_D.itemsize)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("CUDA Error %s" % str(err))

    def free(self):
        if hasattr(self, "destination_buffer"):
            del self.destination_buffer
        if hasattr(self, "source_buffer"):
            del self.source_buffer


class ReductionRT(ExecutableOperation):
    """
    ReductionRT manages the CUTLASS runtime components for reduction
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
  char* ${operation_name}_get_params(${operation_name}${operation_suffix}::Params* params){
    char *bytes = ((char*)(params));
    char *output = new char[sizeof(${operation_name}${operation_suffix}::Params)];
    for (unsigned int i = 0; i < sizeof(${operation_name}${operation_suffix}::Params); i ++)
        output[i] = bytes[i];

    return output;
  }
}
    '''

    def __init__(self, operation: ReductionOperation):
        super().__init__(operation)

        self.operation: ReductionOperation = operation
        self.emitter = EmitReductionInstance('_type')

        self.elements_per_access = self.operation.count
        self.argument_type, self.epilogue_type = get_reduction_params(operation.epilogue_functor)
        self.argtype = [ctypes.POINTER(self.argument_type)]

    def emit(self):
        return self.emitter.emit(self.operation)

    def plan(self, arguments: ReductionArguments):
        block_shape = [self.operation.shape.column(
        ) // self.elements_per_access, self.operation.shape.row(), 1]
        grid_shape = [
            (arguments.problem_size.row + self.operation.shape.row() -
             1) // self.operation.shape.row(),
            (arguments.problem_size.column + self.operation.shape.column() -
                1) // self.operation.shape.column(),
            1
        ]
        return LaunchConfiguration(grid_shape, block_shape, self.shared_memory_capacity)

    def initialize(self):
        err, = cuda.cuFuncSetAttribute(
            self.kernel,
            attrib=cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            value=self.shared_memory_capacity)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))


class ReductionOperation:
    """
    CUTLASS Reduction Operation
    shape: shape of CTA
    outputop: output operator
    r
    """

    def __init__(self, shape: cutlass.MatrixCoord, C: TensorDescription,
                 element_accumulator, element_workspace=None,
                 element_compute=None, epilogue_functor=None,
                 count: int = 1, partitions_per_stage: int = 4) -> None:
        """ Constructor
        """

        self.shape = shape
        #: epilogue functor (default: LinearCombination)
        self.epilogue_functor = epilogue_functor
        #: datatype of accumulator
        self.element_accumulator = element_accumulator

        if element_workspace is None:
            #: datatype of workspace
            self.element_workspace = element_accumulator
        else:
            #: datatype of workspace
            self.element_workspace = element_workspace

        if element_compute is None:
            #: datatype of workspace
            self.element_compute = element_accumulator
        else:
            #: datatype of workspace
            self.element_compute = element_compute

        #: datatype of output
        self.element_output = C.element

        #: operand C
        self.C: TensorDescription = C

        #: reduce op processing size
        self.count: int = count

        #: number of partitions to reduce per stage
        self.partitions_per_stage: int = partitions_per_stage

        self.rt_module: ReductionRT = ReductionRT(self)
        self.argument_type = self.rt_module.argument_type
        self.epilogue_type = self.rt_module.epilogue_type

    #
    def extended_name(self):
        extend_name = "${element_workspace}_${element_accumulator}_${element_compute}_${element_output}"

        return SubstituteTemplate(extend_name,
                                  {
                                      'element_workspace': DataTypeNames[self.element_workspace],
                                      'element_accumulator': DataTypeNames[self.element_accumulator],
                                      'element_compute': DataTypeNames[self.element_compute],
                                      'element_output': DataTypeNames[self.element_output]
                                  })

    #
    def configuration_name(self):
        ''' The full procedural name indicates architecture, extended name, tile size'''

        configuration_name = "cutlass_reduce_split_k_${extended_name}_${threadblock}"

        threadblock = "%dx%d" % (
            self.shape.row(),
            self.shape.column()
        )

        return SubstituteTemplate(
            configuration_name,
            {
                'extended_name': self.extended_name(),
                'threadblock': threadblock
            }
        )

    #
    def procedural_name(self):
        ''' The full procedural name indicates architeture, extended name, tile size'''
        return self.configuration_name()

    def run(self, arguments: ReductionArguments) -> cuda.CUresult:
        """
        Configure and launch the cuda kernel with input arguments
        """
        # get launch configuration
        launch_config = self.rt_module.plan(arguments)

        # get the host and device workspace
        host_workspace = arguments.host_workspace
        device_workspace = None

        # launch the kernel
        err = self.rt_module.run(
            host_workspace, device_workspace, launch_config)

        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('CUDA Error %s' % str(err))

        return err


class EmitReductionInstance:
    def __init__(self, operation_suffix='') -> None:
        self.operation_suffix = operation_suffix
        self.includes = [
            "cutlass/cutlass.h",
            "cutlass/numeric_types.h",
            "cutlass/arch/arch.h",
            "cutlass/arch/mma.h",
            "cutlass/layout/matrix.h",
            "cutlass/gemm/device/gemm.h",
            "cutlass/gemm/device/gemm_universal_adapter.h",
            "cutlass/gemm/kernel/default_gemm_universal.h",
            "cutlass/reduction/kernel/reduce_split_k.h",
            "cutlass/reduction/thread/reduction_operators.h"
        ]
        self.template = """
// Reduction kernel instance
using ${operation_name}_base = 
typename cutlass::reduction::kernel::ReduceSplitK<
  cutlass::MatrixShape<${shape_row}, ${shape_column}>,
  ${epilogue_functor},
  cutlass::reduction::thread::ReduceAdd<
    ${element_accumulator},
    ${element_output},
    ${count}>,
  ${partition_per_stage}>;

struct ${operation_name}${operation_suffix}:
  public ${operation_name}_base { };
      """

    def emit(self, operation: ReductionOperation):

        epilogue_vector_length = int(min(
            operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

        values = {
            'operation_name': operation.configuration_name(),
            'operation_suffix': self.operation_suffix,
            'shape_row': str(operation.shape.row()),
            'shape_column': str(operation.shape.column()),
            'epilogue_functor': operation.epilogue_functor.emit(),
            'element_output': DataTypeTag[operation.element_output],
            'epilogue_vector_length': str(epilogue_vector_length),
            'element_accumulator': DataTypeTag[operation.element_accumulator],
            'element_compute': DataTypeTag[operation.element_compute],
            'element_workspace': DataTypeTag[operation.element_workspace],
            'count': str(operation.count),
            'partition_per_stage': str(operation.partitions_per_stage)
        }

        return SubstituteTemplate(self.template, values)
