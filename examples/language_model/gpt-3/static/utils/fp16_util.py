# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import paddle

# Note(liyurui): The function in this file is copy from
# https://github.com/PaddlePaddle/Paddle/pull/39595.
# Maybe should be remove when Paddle support this.


def cast_model_to_fp16_block(program, amp_lists=None, use_fp16_guard=True):
    from paddle.static.amp import AutoMixedPrecisionLists
    from paddle.fluid.contrib.mixed_precision.fp16_utils import \
        _valid_types, _dtype_to_str, \
        _need_keep_fp32, _insert_cast_op, _keep_fp32_input, _keep_fp32_output

    if amp_lists is None:
        amp_lists = AutoMixedPrecisionLists()
    global_block = program.global_block()
    to_fp16_var_names = set()

    def get_var(block, var_name):
        var = None
        try:
            var = block.var(var_name)
        except ValueError as e:
            print("-- {}, try to get it in the global block --".format(e))
            var = global_block.var(var_name)
            if var is not None:
                print(
                    "-- var {} is got in the global block --".format(var_name))
        return var

    def get_input_output_info(block):
        # A map from output var to ops which generate it.
        output_var_to_ops = collections.defaultdict(list)
        # A map from var to ops which takes it as input.
        input_var_to_ops = collections.defaultdict(list)

        for index, op in enumerate(block.ops):
            for var_name in op.input_arg_names:
                input_var_to_ops[var_name].append([op, index])
            for var_name in op.output_arg_names:
                output_var_to_ops[var_name].append([op, index])
        return output_var_to_ops, input_var_to_ops

    def cast_var(block, op, idx, src_dtype, dest_dtype, var_name):
        num_cast_ops = 0

        var = get_var(block, var_name)
        if var.type not in _valid_types or var.dtype == dest_dtype:
            return num_cast_ops

        assert var.dtype == src_dtype, \
            "The real dtype({}) is not equal to the src dtype({})".format(
                _dtype_to_str(var.dtype), _dtype_to_str(src_dtype))

        cast_name = var_name + '.cast_' + _dtype_to_str(dest_dtype)
        cast_var = block.vars.get(cast_name)
        if cast_var is None or cast_var.dtype != dest_dtype:
            cast_var = block.create_var(name=cast_name,
                                        dtype=dest_dtype,
                                        persistable=False,
                                        stop_gradient=var.stop_gradient)
            block._insert_op_without_sync(idx,
                                          type='cast',
                                          inputs={'X': var},
                                          outputs={'Out': cast_var},
                                          attrs={
                                              'in_dtype': var.dtype,
                                              'out_dtype': cast_var.dtype,
                                              'op_device': op.attr('op_device')
                                          })
            num_cast_ops += 1
        return num_cast_ops, cast_name

    def cast_block(block, parent_fp32var2op=None):
        output_var_to_ops, input_var_to_ops = get_input_output_info(block)

        fp32_var2op = dict()
        if parent_fp32var2op is not None:
            fp32_var2op.update(parent_fp32var2op)
        num_cast_ops = 0
        for idx, op in enumerate(list(block.ops)):
            #if op.type == 'create_py_reader' or op.type == 'read':
            if op.type == 'create_py_reader':
                continue
            if op.has_attr('sub_block'):
                sub_block_id = op.attr('sub_block').id
                cast_block(program.block(sub_block_id), fp32_var2op)
                continue

            if _need_keep_fp32(op, amp_lists.unsupported_list, use_fp16_guard):
                pre_cast_num = _insert_cast_op(block, op, idx + num_cast_ops,
                                               paddle.float16, paddle.float32)
                num_cast_ops += pre_cast_num

                for out_var_name in op.output_arg_names:
                    out_var = block.vars.get(out_var_name)
                    if out_var is None or out_var.type not in _valid_types:
                        continue
                    # FIXME(wangxi): can be float16? multi write?
                    if out_var.dtype == paddle.float16:
                        out_var.desc.set_dtype(paddle.float32)
            else:
                for in_name in op.input_names:
                    # FIXME(wangxi): cast fp16->fp32?
                    if _keep_fp32_input(op, in_name):
                        continue

                    for in_var_name in op.input(in_name):
                        in_var = get_var(block, in_var_name)
                        if in_var is None or in_var.type not in _valid_types:
                            continue

                        if in_var.dtype == paddle.float32 and in_var_name in fp32_var2op:
                            # cast fp32->fp16, rename op input
                            pre_cast_num, cast_name = cast_var(
                                block, fp32_var2op[in_var_name],
                                idx + num_cast_ops, paddle.float32,
                                paddle.float16, in_var_name)
                            op._rename_input(in_var_name, cast_name)
                            num_cast_ops += pre_cast_num
                        elif in_var.dtype == paddle.float32:
                            in_var.desc.set_dtype(paddle.float16)
                            to_fp16_var_names.add(in_var_name)

                        print(
                            "-- op type: {}, in var name: {}, in var dtype: {} --"
                            .format(op.type, in_var_name, in_var.dtype))

                    for out_name in op.output_names:
                        # FIXME(wangxi): can be float16?
                        if _keep_fp32_output(op, out_name):
                            continue

                        for out_var_name in op.output(out_name):
                            out_var = get_var(block, out_var_name)
                            if out_var is None or out_var.type not in _valid_types:
                                continue
                            if out_var.dtype == paddle.float32:
                                out_var.desc.set_dtype(paddle.float16)

                            print(
                                "-- op type: {}, out var name: {}, out var dtype: {} --"
                                .format(op.type, out_var_name, out_var.dtype))

                    if op.has_attr('in_dtype') and op.attr(
                            'in_dtype') == paddle.float32:
                        op._set_attr('in_dtype', paddle.float16)
                    if op.has_attr('out_dtype') and op.attr(
                            'out_dtype') == paddle.float32:
                        op._set_attr('out_dtype', paddle.float16)
                    if op.has_attr('dtype') and op.attr(
                            'dtype') == paddle.float32:
                        op._set_attr('dtype', paddle.float16)

            # record fp32_var to op
            for in_var_name in op.input_arg_names:
                in_var = block.vars.get(in_var_name)
                if in_var is None or in_var.type not in _valid_types:
                    continue
                if in_var.dtype == paddle.float32:
                    fp32_var2op[in_var_name] = op

            for out_var_name in op.output_arg_names:
                out_var = block.vars.get(out_var_name)
                if out_var is None or out_var.type not in _valid_types:
                    continue
                if out_var.dtype == paddle.float32:
                    fp32_var2op[out_var_name] = op

    cast_block(global_block)
    return to_fp16_var_names
