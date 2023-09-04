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

from ast import Num
from audioop import mul
from pipes import Template
import struct
from pycutlass.library import DataTypeTag
from pycutlass import *
import cutlass
from scipy.special import erf

from pycutlass.c_types import MatrixCoord_
from pycutlass.frontend import NumpyFrontend

from cuda import cuda
from cuda import cudart

dtype2ctype = {
    cutlass.float16: ctypes.c_uint16,
    cutlass.float32: ctypes.c_float,
    cutlass.float64: ctypes.c_double,
    cutlass.int32: ctypes.c_int32
}


#################################################################################################
#
# Epilogue Functors
#
#################################################################################################

class EpilogueFunctorBase:
    """
    Base class for thread-level epilogue functors
    """
    def __init__(self) -> None:
        pass
    
    def emit(self, tag, template_argument):
        template = """${tag}<${arguments}>"""
        arguments = ""
        for idx, arg in enumerate(template_argument):
            arguments += arg
            if idx < len(template_argument) - 1:
                arguments += ", "
        values = {
            "tag": tag,
            "arguments": arguments
        }

        return SubstituteTemplate(template, values)
        


class LinearCombination(EpilogueFunctorBase):
    """
    Apply a linear combination operator to an array of elements
    D = alpha * accumulator + beta * source

    :param element_output: data type used to load and store tensors
    
    :param epilogue_vector_length: number of elements computed per operation. 
    Usually it is 128/sizeof_bits<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store

    :param element_accumulator: Accumulator data type

    :param element_epilogue: data type used to compute linear combination
    """
    tag = "cutlass::epilogue::thread::LinearCombination"
    def __init__(
        self, element_output, epilogue_vector_length, 
        element_accumulator=None, element_epilogue=None) -> None: # TODO bind ScaleType
        super().__init__()

        if element_accumulator is None:
            element_accumulator = element_output
        if element_epilogue is None:
            element_epilogue = element_output
        
        self.element_output = element_output
        self.element_accumulator = element_accumulator
        self.element_epilogue = element_epilogue
        self.epilogue_vector_length = epilogue_vector_length

        self.template_arguments = [
            DataTypeTag[element_output], str(epilogue_vector_length),
            DataTypeTag[element_accumulator], DataTypeTag[element_epilogue]
        ]

        # get epilogue output op type
        c_element_epilogue = dtype2ctype[self.element_epilogue]
        element_epilogue = self.element_epilogue

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha_data", ctypes.c_longlong*2),
                ("beta_data", ctypes.c_longlong*2),
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
            ]
            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = element_epilogue(alpha).storage
                self.beta = element_epilogue(beta).storage
        self.epilogue_type = _EpilogueOutputOpParams
    
    def emit(self):
        return super().emit(self.tag, self.template_arguments)


class LinearCombinationClamp(LinearCombination):
    """
    Applies a linear combination operator to an array of elements then clamps 
    the output before converting to the output element type.

    D = alpha * accumulator + beta * source + uniform

    :param element_output: data type used to load and store tensors
    
    :param epilogue_vector_length: number of elements computed per operation. 
    Usually it is 128/sizeof_bits<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store

    :param element_accumulator: Accumulator data type

    :param element_epilogue: data type used to compute linear combination
    """
    tag = "cutlass::epilogue::thread::LinearCombinationClamp"
    def __init__(
        self, element_output, epilogue_vector_length, 
        element_accumulator=None, element_epilogue=None) -> None:
        # Base constructor
        super().__init__(
            element_output, epilogue_vector_length, 
            element_accumulator, element_epilogue)
        
        c_element_epilogue = dtype2ctype[self.element_epilogue]
        element_epilogue = self.element_epilogue

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
            ]
            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = element_epilogue(alpha).storage
                self.beta = element_epilogue(beta).storage
        self.epilogue_type =  _EpilogueOutputOpParams


class FastLinearCombinationClamp(EpilogueFunctorBase):
    """
    Applies a linear combination operator to an array of elements then clamps
    the output before converting to the output element type.

    D = alpha * accumulator + beta * source

    Note: The below method only when problem_size_K <= 256 for signed int8 gemm
    or problem_size_K <= 128 for unsigned int8 gemm. The default approach is
    above.

    :param element_output: data type used to load and store tensors
    
    :param epilogue_vector_length: number of elements computed per operation. 
    Usually it is 128/sizeof_bits<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store
    """
    tag = "cutlass::epilogue::thread::FastLinearCombinationClamp"
    def __init__(self, element_output, epilogue_vector_length, *args) -> None:
        super().__init__()

        self.template_arguments = [
            DataTypeTag[element_output], str(epilogue_vector_length)
        ]

        self.element_accumulator = cutlass.int32
        self.element_epilogue = cutlass.float32

        # get epilogue output op
        c_element_epilogue = dtype2ctype[self.element_epilogue]
        element_epilogue = self.element_epilogue

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
            ]
            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = element_epilogue(alpha).storage
                self.beta = element_epilogue(beta).storage
        self.epilogue_type =  _EpilogueOutputOpParams
    
    def emit(self):
        return super().emit(self.tag, self.template_arguments)


class LinearCombinationGeneric(LinearCombination):
    """
    Applies a linear combination operator followed by an activation function 
    to an array of elements.

    D = activation(alpha * accumulator + beta * source)

    :param activation_functor: input activation functor

    :param element_output: data type used to load and store tensors
    
    :param epilogue_vector_length: number of elements computed per operation. 
    Usually it is 128/sizeof_bits<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store

    :param element_accumulator: Accumulator data type

    :param element_epilogue: data type used to compute linear combination
    """
    tag = "cutlass::epilogue::thread::LinearCombinationGeneric"
    def __init__(
        self, activation_functor,
        element_output, epilogue_vector_length, 
        element_accumulator=None, element_epilogue=None) -> None:
        super().__init__(
            element_output, epilogue_vector_length, 
            element_accumulator, element_epilogue)
        
        self.template_arguments = [
            activation_functor.emit(),] + self.template_arguments
        
        self.activation_functor = activation_functor
        self.element_epilogue = element_epilogue
    
        # get epilogue output op
        self.epilogue_type = self.activation_functor.epilogue_output_op(self.element_epilogue)


class ActivationFunctor:
    """
    Base class for frequently used activation functions
    """
    def __init__(self, element_compute) -> None:
        pass
    @staticmethod
    def numpy(x: np.ndarray):
        raise NotImplementedError()

    def emit(self):
        return self.tag
    
    @staticmethod
    def epilogue_output_op(element_epilogue):
        c_element_epilogue = dtype2ctype[element_epilogue]

        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
            ]
            def __init__(self, alpha, beta, *args) -> None:
                self.alpha = element_epilogue(alpha).storage
                self.beta = element_epilogue(beta).storage
        return _EpilogueOutputOpParams

# identity operator
class identity(ActivationFunctor):
    def numpy(x: np.ndarray):
        return x

# ReLu operator, 
class relu(ActivationFunctor):
    tag = "cutlass::epilogue::thread::ReLu"

    def __init__(self, element_compute):
        super().__init__(element_compute)
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("threshold", dtype2ctype[element_compute])
            ]
            def __init__(self, threshold=0.) -> None:
                self.threshold = element_compute(threshold).storage
        self.argument_type = _Arguments
    
    def emit_visitor(self):
        return "cutlass::ReLUVisitor"
    
    @staticmethod
    def numpy(x: np.ndarray):
        return np.maximum(x, 0)

# Leaky ReLu operator
class leaky_relu(ActivationFunctor):
    tag = "cutlass::epilogue::thread::LeakyReLU"

    def __init__(self, element_compute) -> None:
        super().__init__(element_compute)
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("leaky_alpha", dtype2ctype[element_compute])
            ]
            def __init__(self, leaky_alpha) -> None:
                self.leaky_alpha = element_compute(leaky_alpha).storage
        self.argument_type = _Arguments
    
    def emit_visitor(self):
        return "cutlass::LeakyReLUVisitor"

    @staticmethod
    def numpy(x: np.ndarray, leaky_alpha):
        return np.maximum(x, 0) + np.minimum(x, 0) * leaky_alpha
    
    def epilogue_output_op(self, element_epilogue):
        c_element_epilogue = dtype2ctype[element_epilogue]
        class _EpilogueOutputOpParams(ctypes.Structure):
            _fields_ = [
                ("alpha", c_element_epilogue),
                ("beta", c_element_epilogue),
                ("alpha_ptr", ctypes.c_void_p),
                ("beta_ptr", ctypes.c_void_p),
                ("leaky_alpha", c_element_epilogue)
            ]
            def __init__(self, alpha, beta, leaky_alpha=0.2, *args) -> None:
                self.alpha = element_epilogue(alpha).storage
                self.beta = element_epilogue(beta).storage
                self.alpha_ptr = 0
                self.beta_ptr = 0
                self.leaky_alpha = element_epilogue(leaky_alpha).storage
        return _EpilogueOutputOpParams

# Tanh operator
class tanh(ActivationFunctor):
    tag = "cutlass::epilogue::thread::Tanh"

    def __init__(self, element_compute) -> None:
        super().__init__(element_compute)
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("tmp", ctypes.c_int)
            ]
            def __init__(self, *args) -> None:
                self.tmp = 0
        self.argument_type = _Arguments
    
    def emit_visitor(self):
        return "cutlass::TanhVisitor"

    @staticmethod
    def numpy(x: np.ndarray):
        return np.tanh(x)

def sigmoid_op(x: np.ndarray):
    return 1. / (1. + np.exp(-x))

# Sigmoid operator
class sigmoid(ActivationFunctor):
    tag = "cutlass::epilogue::thread::Sigmoid"

    @staticmethod
    def numpy(x: np.ndarray):
        return sigmoid_op(x)

# SiLu operator
class silu(ActivationFunctor):
    tag = "cutlass::epilogue::thread::SiLu"

    @staticmethod
    def numpy(x: np.ndarray):
        return x * sigmoid_op(x)

# Hardswish operator
class hardswish(ActivationFunctor):
    tag = "cutlass::epilogue::thread::HardSwish"

    @staticmethod
    def numpy(x: np.ndarray):
        relu6 = np.minimum(np.maximum(x + 3., 0), 6.)
        return x * relu6 / 6.

# GELU operator
class gelu(ActivationFunctor):
    tag = "cutlass::epilogue::thread::GELU"

    @staticmethod
    def numpy(x: np.ndarray):
        return 0.5 * x * (1 + erf(x / np.sqrt(2.)))

# reduction operator
def reduction_op(tensor, direction, math, factor):
    batch, m, n = tensor.shape
    if math == "Add":
        if direction == "row":
            num_cta_n = (n + factor - 1) // factor
            reduction = np.transpose(
                np.sum(tensor.reshape(batch, m, num_cta_n, factor), axis=-1), 
                axes=[0, 2, 1]).flatten()
        elif direction == "column":
            num_cta_m = (m + factor - 1) // factor
            reduction = np.sum(
                tensor.reshape(batch, num_cta_m, factor, n), axis=-2).flatten()
        else:
            raise NotImplementedError
        return  reduction
    else:
        raise NotImplementedError

# # GELU operator implemented using the taylor series approximation
# class GELU_taylor(ActivationFunctor):
#     tag = "cutlass::epilogue::thread::GELU_taylor"

# # Computes backwards pass for GELU operator
# class dGELU(ActivationFunctor):
#     tag = "cutlass::epilogue::thread::dGELU"

################################################################################
# Epilogue Visitor
################################################################################


class LayerNorm(EpilogueFunctorBase):
    """
    Apply a linear combination operator to an array of elements
    D = alpha * accumulator + beta * source

    :param element_output: data type used to load and store tensors
    
    :param epilogue_vector_length: number of elements computed per operation. 
    Usually it is 128/sizeof_bits<ElementOutput_>, but we use 64 and 32 sometimes
    when there are not enough data to store

    :param element_accumulator: Accumulator data type

    :param element_epilogue: data type used to compute linear combination
    """
    KernelTemplate = """

cutlass::epilogue::threadblock::EpilogueVisitorLayerNorm<
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    ${operation_name}_default::kThreadCount,
    ${operation_name}_default::Epilogue::OutputTileIterator,
    ${operation_name}_default::Epilogue::AccumulatorFragmentIterator::AccumulatorTile,
    ${element_compute}, // element_compute
    ${element_variance}, // element_variance
    ${element_mean}, // element_mean
    ${element_layer_norm_compute}, // element_layer_norm_compute
    ${epilogue_functor},
    ${shifted_k}>;
"""
    headers = ["gemm/gemm_universal_with_visitor.h",
                "epilogue/epilogue_visitor_with_layernorm.h"]
    def __init__(
        self, elementwise_functor,
        element_variance=None, element_mean=None, 
        element_layer_norm_compute=None, shifted_k=True) -> None: # TODO bind ScaleType
        super().__init__()

        self.elementwise_functor = elementwise_functor
        self.element_compute = elementwise_functor.element_epilogue
        self.element_output = elementwise_functor.element_output
        
        if element_variance is None:
            self.element_variance = self.element_output
        if element_mean is None:
            self.element_mean = self.element_output
        if element_layer_norm_compute is None:
            self.element_layer_norm_compute = self.element_compute
        if shifted_k:
            self.shifted_k = "true"
        else:
            self.shifted_k = "false"
        
        # get epilogue output op
        elementwise_params_type = self.elementwise_functor.epilogue_type
        
        class _EpilogueVisitorParams(ctypes.Structure):
            _fields_ = [
                ("element_wise", elementwise_params_type),
                ("ptr_Variance", ctypes.c_void_p),
                ("ptr_Mean_", ctypes.c_void_p),
                ("ptr_Shifted_K_", ctypes.c_void_p),
                ("extent", MatrixCoord_)
            ]
            def __init__(self, elementwise_params, variance, mean, shift_k, extent) -> None:
                self.element_wise = elementwise_params
                if isinstance(variance, np.ndarray):
                    self.buffer_variance = NumpyFrontend.argument(variance, False)
                    self.buffer_mean = NumpyFrontend.argument(mean, False)
                    self.buffer_shift_k = NumpyFrontend.argument(shift_k, False)
                    self.ptr_Variance = int(self.buffer_variance.ptr)
                    self.ptr_Mean_ = int(self.buffer_mean.ptr)
                    self.ptr_Shifted_K_ = int(self.buffer_shift_k.ptr)
                    self.extent = MatrixCoord_(extent[0], extent[1])

                    self.host_variance = variance
                    self.host_mean = mean
                    self.host_shift_k = shift_k
            
            def sync(self, stream_sync=True):
                if stream_sync:
                    err, = cudart.cudaDeviceSynchronize()
                    if err != cuda.CUresult.CUDA_SUCCESS:
                        raise RuntimeError("CUDA Error %s" % str(err))
                
                # if hasattr(self, "host_variance"):
                err, = cuda.cuMemcpyDtoH(
                    self.host_variance, cuda.CUdeviceptr(self.ptr_Variance), 
                    self.host_variance.size * self.host_variance.itemsize)
                err, = cuda.cuMemcpyDtoH(
                    self.host_mean, cuda.CUdeviceptr(self.ptr_Mean_), 
                    self.host_mean.size * self.host_mean.itemsize)
                err, = cuda.cuMemcpyDtoH(
                    self.host_shift_k, cuda.CUdeviceptr(self.ptr_Shifted_K_), 
                    self.host_shift_k.size * self.host_shift_k.itemsize)
                if err != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError("CUDA Error %s" % str(err))

        self.epilogue_type =  _EpilogueVisitorParams

    def emit(self, operation):
        values = {
            'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
            'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
            'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
            'operation_name': operation.procedural_name(),
            'element_compute': DataTypeTag[self.element_compute],
            'element_variance': DataTypeTag[self.element_variance],
            'element_mean': DataTypeTag[self.element_mean],
            'element_layer_norm_compute': DataTypeTag[self.element_layer_norm_compute],
            'epilogue_functor': self.elementwise_functor.emit(),
            'shifted_k': self.shifted_k
        }
        return SubstituteTemplate(self.KernelTemplate, values)



class AccumulatorOp:
    Template = """
using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpAccumulator<${element_accumulator}, ${elements_per_access}>;
"""
    counter = 0
    def __init__(self, element_accumulator, elements_per_access) -> None:
        self.element_accumulator = element_accumulator
        self.elements_per_access = elements_per_access

        self.instance_name = "AccumulatorOp%d" % AccumulatorOp.counter
        AccumulatorOp.counter += 1


        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("tmp", ctypes.c_int)
            ]
            def __init__(self):
                self.tmp = 0
        
        self.argument_type = _Arguments
    
    def emit(self, *args):
        values = {
            "instance_name": self.instance_name,
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "elements_per_access": str(self.elements_per_access)
        }
        return SubstituteTemplate(self.Template, values)


class LinearCombinationOp:
    Template = """
${visitor_a}

${visitor_b}

using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpLinearCombination<
    ${element_accumulator}, ${element_compute}, 
    ${elements_per_access}, ${visitor_a_name}, ${visitor_b_name}>;
"""
    counter = 0
    def __init__(self, element_accumulator, element_compute, 
        elements_per_access, visitor_a, visitor_b) -> None:
        #
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
        self.visitor_a = visitor_a
        self.visitor_b = visitor_b
        
        self.instance_name = "LinearCombinationOp%d" % LinearCombinationOp.counter
        LinearCombinationOp.counter += 1

        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("alpha", dtype2ctype[self.element_compute]),
                ("beta", dtype2ctype[self.element_compute]),
                ("visitor_a", self.visitor_a.argument_type),
                ("visitor_b", self.visitor_b.argument_type)
            ]
            def __init__(self, alpha, beta, visitor_a_arg, visitor_b_arg) -> None:
                self.alpha = element_compute(alpha).storage
                self.beta = element_compute(beta).storage
                self.visitor_a = visitor_a_arg
                self.visitor_b = visitor_b_arg
        
        self.argument_type = _Arguments
    
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "element_compute": DataTypeTag[self.element_compute],
            "elements_per_access": str(self.elements_per_access),
            "visitor_a_name": self.visitor_a.instance_name,
            "visitor_b_name": self.visitor_b.instance_name,
            "visitor_a": self.visitor_a.emit(operation),
            "visitor_b": self.visitor_b.emit(operation)
        }
        return SubstituteTemplate(self.Template, values)

class VectorAdd:
    def __init__(self, *args) -> None:
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("tmp", ctypes.c_int)
            ]
            def __init__(self, *args) -> None:
                self.tmp = 0
        self.argument_type = _Arguments

    def emit(self):
        return "cutlass::VectorAdd"

class VectorMult:
    def __init__(self, *args) -> None:
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("tmp", ctypes.c_int)
            ]
            def __init__(self, *args) -> None:
                self.tmp = 0
        self.argument_type = _Arguments

    def emit(self):
        return "cutlass::VectorMult"
        

class BinaryOp:
    Template = """
${visitor_a}

${visitor_b}

using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpBinary<
    ${element_accumulator}, ${element_compute}, 
    ${elements_per_access}, ${visitor_a_name}, ${visitor_b_name}, ${binary_op}>;
"""
    counter = 0
    def __init__(self, element_accumulator, element_compute, 
        elements_per_access, visitor_a, visitor_b, binary_op) -> None:
        #
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
        self.visitor_a = visitor_a
        self.visitor_b = visitor_b
        self.binary_op = binary_op

        self.instance_name = "BinaryOp%d" % BinaryOp.counter
        BinaryOp.counter += 1

        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("binary_param", binary_op.argument_type),
                ("visitor_a", self.visitor_a.argument_type),
                ("visitor_b", self.visitor_b.argument_type)
            ]
            def __init__(self, binary_param, visitor_a_arg, visitor_b_arg) -> None:
                self.binary_param = binary_param
                self.visitor_a = visitor_a_arg
                self.visitor_b = visitor_b_arg
        
        self.argument_type = _Arguments
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "element_compute": DataTypeTag[self.element_compute],
            "elements_per_access": str(self.elements_per_access),
            "visitor_a_name": self.visitor_a.instance_name,
            "visitor_b_name": self.visitor_b.instance_name,
            "visitor_a": self.visitor_a.emit(operation),
            "visitor_b": self.visitor_b.emit(operation),
            "binary_op": self.binary_op.emit()
        }
        return SubstituteTemplate(self.Template, values)


class Mult:
    def __init__(self, element_compute) -> None:
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("alpha", dtype2ctype[element_compute])
            ]
            def __init__(self, alpha) -> None:
                self.alpha = element_compute(alpha).storage
        
        self.argument_type = _Arguments
    
    def emit_visitor(self):
        return "cutlass::Mult"

class UnaryOp:
    Template = """
${visitor}

using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpUnary<
    ${element_accumulator}, ${element_compute},
    ${elements_per_access}, ${visitor_name}, ${unary_op}>;
"""
    counter = 0
    def __init__(self, element_accumulator, element_compute,
        elements_per_access, visitor, unary_op) -> None:
        #
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
        self.visitor = visitor
        self.unary_op = unary_op

        self.instance_name = "UnaryOp%d" % UnaryOp.counter
        UnaryOp.counter += 1

        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("unary_param", unary_op.argument_type),
                ("visitor_arg", self.visitor.argument_type)
            ]
            def __init__(self, unary_param, visitor_arg) -> None:
                self.unary_param = unary_param
                self.visitor_arg = visitor_arg
        
        self.argument_type = _Arguments
    
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "element_compute": DataTypeTag[self.element_compute],
            "elements_per_access": str(self.elements_per_access),
            "visitor_name": self.visitor.instance_name,
            "unary_op": self.unary_op.emit_visitor(),
            "visitor": self.visitor.emit(operation)
        }
        return SubstituteTemplate(self.Template, values)



class RowBroadcastOp:
    Template = """
using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpRowBroadcast<
    ${element_accumulator}, ${element_fragment}, ${input_tile_iterator}>;
"""
    counter = 0
    def __init__(self, element_accumulator, element_fragment) -> None:
        self.element_accumulator = element_accumulator
        self.element_fragment = element_fragment

        self.instance_name = "RowBroadcastOp%d" % RowBroadcastOp.counter
        RowBroadcastOp.counter += 1
        
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("broadcast_ptr", ctypes.c_void_p),
                ("batch_stride", ctypes.c_longlong)
            ]
            def __init__(self, broadcast_ptr, batch_stride=0):
                self.broadcast_ptr = int(broadcast_ptr)
                self.batch_stride = batch_stride
        
        self.argument_type = _Arguments
    
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "element_fragment": DataTypeTag[self.element_fragment],
            "input_tile_iterator": operation.procedural_name() + "_default::Epilogue::OutputTileIterator"
        }
        return SubstituteTemplate(self.Template, values)


class ColumnBroadcastOp:
    Template = """
using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpColumnBroadcast<
    ${element_accumulator}, ${element_fragment}, ${input_tile_iterator}>;
"""
    counter = 0
    def __init__(self, element_accumulator, element_fragment) -> None:
        self.element_accumulator = element_accumulator
        self.element_fragment = element_fragment

        self.instance_name = "ColumnBroadcastOp%d" % ColumnBroadcastOp.counter
        ColumnBroadcastOp.counter += 1
        
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("broadcast_ptr", ctypes.c_void_p),
                ("batch_stride", ctypes.c_longlong)
            ]
            def __init__(self, broadcast_ptr, batch_stride=0):
                self.broadcast_ptr = int(broadcast_ptr)
                self.batch_stride = batch_stride
        
        self.argument_type = _Arguments
    
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "element_fragment": DataTypeTag[self.element_fragment],
            "input_tile_iterator": operation.procedural_name() + "_default::Epilogue::OutputTileIterator"
        }
        return SubstituteTemplate(self.Template, values)


class TensorInputOp:
    Template = """
using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpTensorInput<
    ${element_accumulator}, ${input_tile_iterator}>;
"""
    counter = 0
    def __init__(self, element_accumulator) -> None:
        self.element_accumulator = element_accumulator

        self.instance_name = "TensorInputOp%d" % TensorInputOp.counter
        TensorInputOp.counter += 1
        
        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("input_ptr", ctypes.c_void_p),
                ("ldt", ctypes.c_int),
                ("batch_stride", ctypes.c_longlong)
            ]
            def __init__(self, input_ptr, ldt, batch_stride=0) -> None:
                self.input_ptr = int(input_ptr)
                self.ldt = ldt
                self.batch_stride = batch_stride
        
        self.argument_type = _Arguments
    
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "input_tile_iterator": operation.procedural_name() + "_default::Epilogue::OutputTileIterator"
        }
        return SubstituteTemplate(self.Template, values)

class TensorOutputOp:
    Template = """
${visitor}

using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpTensorOutput<
    ${element_accumulator}, ${output_tile_iterator}, ${visitor_name}>;
"""
    counter = 0
    def __init__(self, element_accumulator, visitor) -> None:
        self.element_accumulator = element_accumulator
        self.visitor = visitor

        self.instance_name = "TensorOutputOp%d" % TensorOutputOp.counter
        TensorOutputOp.counter += 1

        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("output_ptr", ctypes.c_void_p),
                ("ldt", ctypes.c_int),
                ("batch_stride", ctypes.c_longlong),
                ("visitor_arg", self.visitor.argument_type)
            ]
            def __init__(self, output_ptr, ldt, visitor_arg, batch_stride=0) -> None:
                self.output_ptr = int(output_ptr)
                self.ldt = int(ldt)
                self.visitor_arg = visitor_arg
                self.batch_stride = batch_stride
        
        self.argument_type = _Arguments
    
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "output_tile_iterator": operation.procedural_name() + "_default::Epilogue::OutputTileIterator",
            "visitor_name": self.visitor.instance_name,
            "visitor": self.visitor.emit(operation)
        }
        return SubstituteTemplate(self.Template, values)


class ColumnReductionOp:
    Template = """
${visitor}

using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpColumnReduction<
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    ${element_accumulator}, ${element_reduction}, ${element_reduction_accumulator},
    ${output_tile_iterator}, ${visitor_name}>;
"""
    counter = 0
    def __init__(self, element_accumulator, element_reduction, 
        element_reduction_accumulator, visitor) -> None:
        self.element_accumulator = element_accumulator
        self.element_reduction = element_reduction
        self.element_reduction_accumulator = element_reduction_accumulator
        self.visitor = visitor

        self.instance_name = "ColumnReductionOp%d" % ColumnReductionOp.counter
        ColumnReductionOp.counter += 1

        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("reduction_ptr", ctypes.c_void_p),
                ("batch_stride", ctypes.c_longlong),
                ("visitor_arg", self.visitor.argument_type)
            ]
            def __init__(self, reduction_ptr, visitor_arg, batch_stride=0) -> None:
                self.reduction_ptr = reduction_ptr
                self.batch_stride = batch_stride
                self.visitor_arg = visitor_arg
        
        self.argument_type = _Arguments
    
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
            'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
            'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "element_reduction": DataTypeTag[self.element_reduction],
            "element_reduction_accumulator": DataTypeTag[self.element_reduction_accumulator],
            "output_tile_iterator": operation.procedural_name() + "_default::Epilogue::OutputTileIterator",
            "visitor_name": self.visitor.instance_name,
            "visitor": self.visitor.emit(operation)
        }
        return SubstituteTemplate(self.Template, values)


class RowReductionOp:
    Template = """
${visitor}

using ${instance_name} = cutlass::epilogue::threadblock::VisitorOpRowReduction<
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    ${element_accumulator}, ${element_reduction}, ${element_reduction_accumulator},
    ${output_tile_iterator}, ${visitor_name}>;
"""
    counter = 0
    def __init__(self, element_accumulator, element_reduction, 
        element_reduction_accumulator, visitor) -> None:
        self.element_accumulator = element_accumulator
        self.element_reduction = element_reduction
        self.element_reduction_accumulator = element_reduction_accumulator
        self.visitor = visitor

        self.instance_name = "RowReductionOp%d" % RowReductionOp.counter
        RowReductionOp.counter += 1

        class _Arguments(ctypes.Structure):
            _fields_ = [
                ("reduction_ptr", ctypes.c_void_p),
                ("batch_stride", ctypes.c_longlong),
                ("visitor_arg", self.visitor.argument_type)
            ]
            def __init__(self, reduction_ptr, visitor_arg, batch_stride=0) -> None:
                self.reduction_ptr = reduction_ptr
                self.visitor_arg = visitor_arg
                self.batch_stride = batch_stride
        
        self.argument_type = _Arguments
    
    def emit(self, operation):
        values = {
            "instance_name": self.instance_name,
            'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
            'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
            'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
            "element_accumulator": DataTypeTag[self.element_accumulator],
            "element_reduction": DataTypeTag[self.element_reduction],
            "element_reduction_accumulator": DataTypeTag[self.element_reduction_accumulator],
            "output_tile_iterator": operation.procedural_name() + "_default::Epilogue::OutputTileIterator",
            "visitor_name": self.visitor.instance_name,
            "visitor": self.visitor.emit(operation)
        }
        return SubstituteTemplate(self.Template, values)
