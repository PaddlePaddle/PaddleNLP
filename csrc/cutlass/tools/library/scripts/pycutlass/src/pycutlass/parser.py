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

from typing import Generic, TypeVar
from treelib import Tree
import numpy as np

from pycutlass import *
import pycutlass

import ast
import textwrap
import inspect

################################################################################
# Type annotation for input arguments
################################################################################

Ttype = TypeVar("Ttype")
Dtype = TypeVar("Dtype")

class NDArray(np.ndarray, Generic[Ttype, Dtype]):
    pass

################################################################################
# Operations
################################################################################

operators = {
    ast.Add: "Add",
    ast.Div: "Div",
    ast.Eq: "Equal",
    ast.Mult: "Mult"
}

################################################################################
# AST Node abstractions
################################################################################
class UnaryNode:
    cnt = 0
    # Concept: this is created by the BinOp Node in python ast
    def __init__(self, 
        element_accumulator, element_compute, elements_per_access,
        node, args) -> None:
        if isinstance(node, BinOpNode):
            self.op = node.op
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                self.op = node.func.id
            elif isinstance(node.func, ast.Attribute):
                self.op = node.func.value.id
            else:
                raise TypeError
        else:
            raise TypeError
        self.tag = "Unary" + self.op + str(UnaryNode.cnt)
        self.id = self.op + str(UnaryNode.cnt)
        self.args = args
        UnaryNode.cnt += 1

        self.type = "tensor"

        self.epilogue_op = getattr(pycutlass, self.op)(element_compute)

        # data types
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
    
    def get_epilogue_node(self, visitors):
        self.epilogue_node = UnaryOp(
            self.element_accumulator, self.element_compute, 
            self.elements_per_access, *visitors, self.epilogue_op)
    
    def get_argument(self, visitor_args, kwargs):
        epilogue_ops = []
        for arg in self.args:
            try:
                epilogue_ops.append(kwargs[arg])
            except:
                epilogue_ops.append(arg) # direct arguments like constant
        self.argument = self.epilogue_node.argument_type(self.epilogue_op.argument_type(*epilogue_ops), *visitor_args)


class BinOpNode:
    cnt = 0
    # Concept: this is created by the BinOp Node in python ast
    def __init__(self, 
        element_accumulator, element_compute, elements_per_access,
        node) -> None:
        self.op = operators[type(node.op)]
        self.tag = "Binary" + self.op + str(BinOpNode.cnt)
        self.id = self.op + str(BinOpNode.cnt)
        self.args = None
        BinOpNode.cnt += 1

        self.type = "tensor"

        self.epilogue_op = getattr(pycutlass, "Vector"+self.op)(element_compute)

        # data types
        self.element_accumulator = element_accumulator
        self.element_compute = element_compute
        self.elements_per_access = elements_per_access
    
    def get_epilogue_node(self, visitors):
        self.epilogue_node = BinaryOp(
            self.element_accumulator, self.element_compute, 
            self.elements_per_access, *visitors, self.epilogue_op)
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(self.epilogue_op.argument_type(self.args), *visitor_args)


class NameNode:
    # Concept: this is created by the Name Node in python ast
    def __init__(self, node) -> None:
        try:
            self.id = node.id
        except:
            self.id = node.targets[0].id
        self.tag = self.id

class ScalarInputNode(NameNode):
    # Concept: scalar
    def __init__(self, node) -> None:
        super().__init__(node)
        self.tag = "Scalar:" + self.tag
        self.type = "scalar"

class AccumulatorNode(NameNode):
    # Concept: VisitorOpAccumulator
    def __init__(self, 
        element_accumulator, elements_per_access, node) -> None:
        super().__init__(node)
        self.tag = "Accum:" + self.tag
        self.type = "tensor"

        self.element_accumulator = element_accumulator
        self.elements_per_access = elements_per_access

    def get_epilogue_node(self, visitors):
        self.epilogue_node = AccumulatorOp(
            self.element_accumulator, self.elements_per_access)
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type()

class TensorInputNode(NameNode):
    # Concept: VisitorOpTensorInput
    def __init__(self, element_accumulator, node) -> None:
        super().__init__(node)
        self.tag = "TensorInput:" + self.tag
        self.type = "tensor"
        self.element_accumulator = element_accumulator
    
    def get_epilogue_node(self, *args):
        self.epilogue_node = TensorInputOp(self.element_accumulator)
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(
            kwargs[self.id + "_ptr"], kwargs["problem_size"][1], 
            kwargs["problem_size"][0] * kwargs["problem_size"][1])

class RowBroadcastNode(NameNode):
    # Concept: VisitorOpRowBroadcast
    def __init__(self, element_accumulator, element_fragment, node) -> None:
        super().__init__(node)
        #
        self.tag = "RowBroadcast:" + self.tag
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.element_fragment = element_fragment
    
    def get_epilogue_node(self, *args):
        self.epilogue_node = RowBroadcastOp(
            self.element_accumulator, self.element_fragment)
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(kwargs[self.id + "_ptr"], kwargs["problem_size"][1])

class ColumnBroadcastNode(NameNode):
    # Concept: VisitorOpColumnBroadcast
    def __init__(self, element_accumulator, element_fragment, node) -> None:
        super().__init__(node)
        self.tag = "ColumnBroadcast:" + self.tag
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.element_fragment = element_fragment
    
    def get_epilogue_node(self, *args):
        self.epilogue_node = ColumnBroadcastOp(
            self.element_accumulator, self.element_fragment)
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(kwargs[self.id + "_ptr"], kwargs["problem_size"][0])

class TensorOutputNode(NameNode):
    # Concept: VisitorOpTensorOutput
    def __init__(self, element_accumulator, node) -> None:
        super().__init__(node)
        self.tag = "TensorOutput:" + self.tag
        self.type = "tensor"
        self.element_accumulator = element_accumulator

    def get_epilogue_node(self, visitors):
        self.epilogue_node = TensorOutputOp(self.element_accumulator, *visitors)
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(kwargs[self.id + "_ptr"], kwargs["problem_size"][1], *visitor_args, kwargs["problem_size"][0] * kwargs["problem_size"][1])

class RowReductionNode:
    # Concept: RowReductionOp
    def __init__(self, element_accumulator, element_reduction,
        element_reduction_accumulator, id, factor) -> None:
        #
        self.id = id
        self.tag = "RowReduction:" + self.id
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.element_reduction = element_reduction
        self.element_reduction_accumulator = element_reduction_accumulator
        self.factor = factor
    
    def get_epilogue_node(self, visitors):
        self.epilogue_node = RowReductionOp(
            self.element_accumulator, self.element_reduction, 
            self.element_reduction_accumulator, *visitors)
    
    def get_batch_stride(self, problem_size):
        return problem_size[0] * ((problem_size[1] + self.factor - 1) // self.factor)
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(kwargs[self.id + "_ptr"], *visitor_args, self.get_batch_stride(kwargs["problem_size"]))

class ColumnReductionNode:
    # Concept: ColumnReductionOp
    def __init__(self, element_accumulator, element_reduction,
        element_reduction_accumulator, id, factor) -> None:
        #
        self.id = id
        self.tag = "ColumnReduction:" + self.id
        self.type = "tensor"
        self.element_accumulator = element_accumulator
        self.element_reduction = element_reduction
        self.element_reduction_accumulator = element_reduction_accumulator
        self.factor = factor
    
    def get_epilogue_node(self, visitors):
        self.epilogue_node = ColumnReductionOp(
            self.element_accumulator, self.element_reduction, 
            self.element_reduction_accumulator, *visitors)
    
    def get_batch_stride(self, problem_size):
        return problem_size[1] * ((problem_size[0] + self.factor - 1) // self.factor)
    
    def get_argument(self, visitor_args, kwargs):
        self.argument = self.epilogue_node.argument_type(kwargs[self.id + '_ptr'], *visitor_args, self.get_batch_stride(kwargs["problem_size"]))

################################################################################
# Epilogue parser function
################################################################################
class EpilogueAST(ast.NodeVisitor):
    def __init__(self, epilogue, 
        tile_description,
        element_accumulator, elements_per_access, 
        element_compute, element_output) -> None:
        #
        
        self.tile_description = tile_description
        self.element_accumulator = element_accumulator
        self.elements_per_access = elements_per_access
        self.element_compute = element_compute
        self.element_output = element_output
        self.epilogue = epilogue

        self.source = textwrap.dedent(inspect.getsource(epilogue.__call__))
        self.ast_tree = ast.parse(self.source)
        self.epilogue_tree = Tree()

        
        # print(ast.dump(self.ast_tree, indent=4)) # For Debug purpose

        # input arguments
        self.input_args = {}
        # return nodes
        self.returns = []
        # reduction source nodes
        self.reduction_source = {}

        # stack used to keep the parent node id
        self.stack = []

        # visit the AST
        self.visit(self.ast_tree)

    # visit the name node
    def visit_Name(self, node):
        # append the return ids into self.returns
        if self.stack[-1] == "return":
            self.returns.append(node.id)
        else:
            # accum is produced from accumulator node
            if node.id == "accum":
                name_node = AccumulatorNode(
                    self.element_accumulator, self.elements_per_access, node)
            else:
                # for input nodes
                if node.id in self.input_args.keys():
                    type = self.input_args[node.id][0]
                    if type == "tensor":
                        name_node = TensorInputNode(self.element_accumulator, node)
                    elif type == "row":
                        name_node = RowBroadcastNode(self.element_accumulator, self.element_compute, node)
                    elif type == "column":
                        name_node = ColumnBroadcastNode(self.element_accumulator, self.element_compute, node)
                    elif type == "scalar":
                        name_node = ScalarInputNode(node)
                    else:
                        raise ValueError(type)
                # for output nodes
                else:
                    name_node = TensorOutputNode(self.element_accumulator, node)
            self.epilogue_tree.create_node(name_node.tag, name_node.id, data=name_node, parent=self.stack[-1])
    
    def visit_Assign(self, node):
        pre_assign_node = self.epilogue_tree.get_node(node.targets[0].id)
        if pre_assign_node is None:
            # The assign is to a root node
            # skip the reduction nodes
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    func_type = node.value.func.id
                elif isinstance(node.value.func, ast.Attribute):
                    func_type = node.value.func.value.id
                else:
                    raise TypeError
                if func_type == 'reduction_op':
                    self.reduction_source[node.value.args[0].id] = [node.value.args[1].value, node.value.args[2].value, node.targets[0].id]
                    return
            name_node = TensorOutputNode(self.element_accumulator, node)
            self.epilogue_tree.create_node(name_node.tag, name_node.id, data=name_node)
            self.stack.append(name_node.id)
        else:
            if node.targets[0].id in self.returns or node.targets[0].id in self.reduction_source.keys():
                self.stack.append(node.targets[0].id)
            else:
                self.stack.append(pre_assign_node.predecessor(self.epilogue_tree.identifier))
                self.epilogue_tree.remove_node(node.targets[0].id)
        
        # get child tag
        self.visit(node.value)
        self.stack.pop()
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_type = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_type = node.func.value.id
        else:
            raise TypeError
        if func_type == "reduction_op":
            self.visit(node.args[0])
        else:
            arg_list = []
            for idx, arg in enumerate(node.args):
                if idx == 0: continue
                if isinstance(arg, ast.Constant):
                    arg_list.append(arg.value)
                elif isinstance(arg, ast.Name):
                    arg_list.append(arg.id)
                else:
                    raise TypeError

            unary_node = UnaryNode(self.element_accumulator, self.element_compute, self.elements_per_access, node, arg_list)
            self.epilogue_tree.create_node(unary_node.tag, unary_node.id, parent=self.stack[-1], data=unary_node)
            self.stack.append(unary_node.id)
            self.visit(node.args[0])
            self.stack.pop()
    
    def visit_BinOp(self, node):
        binop = BinOpNode(self.element_accumulator, self.element_compute,
                    self.elements_per_access, node)
        self.epilogue_tree.create_node(binop.tag, binop.id, data=binop, parent=self.stack[-1])
        self.stack.append(binop.id)
        self.visit(node.left)
        self.visit(node.right)
        self.stack.pop()
    
    def visit_Return(self, node):
        self.stack.append("return")
        self.visit(node.value)
        self.stack.pop()
    
    # # A function definition
    def visit_FunctionDef(self, node: ast.FunctionDef):
        # visit args
        for arg in node.args.args:
            if arg.arg == "self": continue
            if isinstance(arg.annotation, ast.Constant):
                self.input_args[arg.arg] = [arg.annotation.value, ]
        # visit the assign in the reverse order
        for idx in range(len(node.body)):
            self.visit(node.body[-1-idx])
    
    #
    # Tree optimization pass
    #

    # pass 1: lower Binary to Unary
    def pass_binary_2_unary(self, tree, nid):
        node = tree.get_node(nid)
        if isinstance(node.data, BinOpNode):
            lhs_node = tree.get_node(node.successors(tree.identifier)[0])
            left_type = lhs_node.data.type
            rhs_node = tree.get_node(node.successors(tree.identifier)[1])
            right_type = rhs_node.data.type

            if left_type == "scalar" and right_type == "tensor":
                node.data = UnaryNode(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access,
                    node.data, [lhs_node.data.id,])
                node.tag = node.data.tag
                tree.remove_node(lhs_node.data.id)
                self.pass_binary_2_unary(tree, rhs_node.data.id)
            
            elif left_type == "tensor" and right_type == "scalar":
                node.data = UnaryNode(
                    self.element_accumulator, self.element_compute,
                    self.elements_per_access,
                    node.data, [rhs_node.id,])
                node.tag = node.data.tag
                tree.remove_node(rhs_node.data.id)
                self.pass_binary_2_unary(tree, lhs_node.data.id)
            
            else:
                self.pass_binary_2_unary(tree, lhs_node.data.id)
                self.pass_binary_2_unary(tree, rhs_node.data.id)
        else:
            for child in node.successors(tree.identifier):
                self.pass_binary_2_unary(tree, child)
    
    # pass 2: inject reduction nodes
    def pass_inject_reduction(self, tree, nid):
        node = tree.get_node(nid)
        if isinstance(node.data, TensorOutputNode):
            if node.data.id in self.reduction_source.keys():
                direction = self.reduction_source[node.data.id][0]
                target = self.reduction_source[node.data.id][-1]
                if direction == 'row':
                    reduction_node = RowReductionNode(
                        self.element_accumulator, self.element_output,
                        self.element_accumulator, target, self.tile_description.threadblock_shape[1])
                elif direction == "column":
                    reduction_node = ColumnReductionNode(
                        self.element_accumulator, self.element_output,
                        self.element_accumulator, target, self.tile_description.threadblock_shape[0])
                else:
                    raise ValueError(direction)
                child_nid = node.successors(tree.identifier)[0]
                # if this output node is injected only for reduction
                if node.data.id not in self.returns:
                    # get reduction config from disc
                    node.data = reduction_node
                    node.tag = reduction_node.tag
                    self.pass_inject_reduction(tree, child_nid)
                # if this output node is also a tensor output, inject reduction as its children
                else:
                    # get child node
                    tree.create_node(reduction_node.tag, reduction_node.id, data=reduction_node, parent=node.data.id)
                    tree.move_node(child_nid, reduction_node.id)
                    child = tree.get_node(child_nid)
                    for grand_child in child.successors(tree.identifier):
                        self.pass_inject_reduction(tree, grand_child)
            else:
                for child in node.successors(tree.identifier):
                    self.pass_inject_reduction(tree, child)
        else:
            for child in node.successors(tree.identifier):
                self.pass_inject_reduction(tree, child)

    def pass_inject_epilogue_op(self, tree, nid):
        node = tree.get_node(nid)
        visitors = []
        for child in node.successors(tree.identifier):
            visitors.append(self.pass_inject_epilogue_op(tree, child))
        
        node.data.get_epilogue_node(visitors)
        return node.data.epilogue_node

    def get_arguments(self, tree, nid, kwargs):
        node = tree.get_node(nid)
        visitor_args = []
        for child in node.successors(tree.identifier):
            visitor_args.append(self.get_arguments(tree, child, kwargs))
        
        node.data.get_argument(visitor_args, kwargs)
        return node.data.argument

class EpilogueVisitTree:
    KernelTemplate = """
${visitor}

using ${operation_name}_EpilogueVisitor = cutlass::epilogue::threadblock::EpilogueVisitorGeneric<${visitor_name}>;
""" 
    def __init__(self, elementwise_functor, tile_description,
        element_accumulator, elements_per_access, 
        element_compute, element_output) -> None:
        #
        # data types
        self.tile_description = tile_description
        self.element_accumulator = element_accumulator
        self.elements_per_access = elements_per_access
        self.element_compute = element_compute
        self.element_output = element_output
        self.elementwise_functor = elementwise_functor
        pass
    
    def initialize(self):
        function = EpilogueAST(self, self.tile_description,
            self.element_accumulator, self.elements_per_access,
            self.element_compute, self.element_output)
        #
        tree = function.epilogue_tree
        self.tree = tree
        function.pass_binary_2_unary(self.tree, self.tree.root)
        function.pass_inject_reduction(self.tree, self.tree.root)
        function.pass_inject_epilogue_op(self.tree,self.tree.root)

        visitor = self.tree.get_node(self.tree.root).data.epilogue_node
        self.visitor = visitor

        class _Argument(ctypes.Structure):
            _fields_ = [
                ("visitor_arg", visitor.argument_type)
            ]
            def __init__(self, **kwargs) -> None:
                # process input args
                _kwargs = {}
                for input_key in function.input_args.keys():
                    if input_key == "accum":
                        continue
                    if function.input_args[input_key][0] == "scalar": 
                        continue
                    # tensor input
                    else:
                        setattr(self, "buffer_tensor_" + input_key, NumpyFrontend.argument(kwargs[input_key], False))
                        setattr(self, input_key + "_ptr", int(getattr(self, "buffer_tensor_" + input_key).ptr))
                        _kwargs[input_key+"_ptr"] = getattr(self, input_key + "_ptr")
                # process the return args
                for ret in function.returns:
                    setattr(self, "buffer_tensor_" + ret, NumpyFrontend.argument(kwargs[ret], True))
                    setattr(self, ret + "_ptr", int(getattr(self, "buffer_tensor_" + ret).ptr))
                    _kwargs[ret+"_ptr"] = getattr(self, ret + "_ptr")
                    setattr(self, "host_tensor_" + ret, kwargs[ret])
                
                _kwargs.update(kwargs)
                function.get_arguments(tree, tree.root, _kwargs)
                self.visitor_arg = tree.get_node(tree.root).data.argument
            
            def sync(self, stream_sync=True):
                if stream_sync:
                    err, = cudart.cudaDeviceSynchronize()
                    if err != cuda.CUresult.CUDA_SUCCESS:
                        raise RuntimeError("CUDA Error %s" % str(err))
                
                for ret in function.returns:
                    err, = cuda.cuMemcpyDtoH(
                        getattr(self, "host_tensor_" + ret), cuda.CUdeviceptr(getattr(self, ret + "_ptr")),
                        getattr(self, "host_tensor_" + ret).size * getattr(self, "host_tensor_" + ret).itemsize
                    )
                if err != cuda.CUresult.CUDA_SUCCESS:
                    raise RuntimeError("CUDA Error %s" % str(err))
                pass
        
        self.epilogue_type = _Argument
    
    def emit(self, operation):
        values = {
            'visitor': self.visitor.emit(operation),
            'operation_name': operation.procedural_name(),
            'visitor_name': self.visitor.instance_name
        }
        return SubstituteTemplate(self.KernelTemplate, values)
