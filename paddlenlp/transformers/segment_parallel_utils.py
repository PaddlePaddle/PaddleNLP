# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet import fleet

sys.path.append("../../")


# llama attention module in paddlenlp/transformers/llama/modeling.py-LlamaAttention
# gpt model_zoo/gpt-3/ppfleetx/models/language_model/gpt/dygraph/hybrid_model.py-MultiHeadAttention

def save_tensor(input_data, save_dir="dp_input_data", suffix_name="", save_once=False, save_as_numpy=False):
    os.makedirs(save_dir, exist_ok=True)
    dp_rank = 0
    sep_rank = 0
    try:
        _hcg = fleet.get_hybrid_communicate_group()
        dp_rank = _hcg.get_data_parallel_rank()
        sep_rank = _hcg.get_sep_parallel_rank()
        save_dir="sep_input_data"
    except:
        pass
    def unique_name(file_path):
        i = -1
        prefix = file_path.split(".")[0]
        suffix = file_path.split(".")[1]
        while os.path.exists(file_path):
            i += 1
            file_path = f"{prefix}_{i}.{suffix}"
        print(f"unique_name file_path:{file_path}")
        return file_path

    def save_path_name(dp_rank, sep_rank, param_name=""):
        suffix = "npy" if save_as_numpy else "pd"
        if param_name == "":
            return f"{save_dir}/dp{dp_rank}_sep{sep_rank}_{suffix_name}.{suffix}"
        else:
            return f"{save_dir}/dp{dp_rank}_sep{sep_rank}_{suffix_name}_{param_name}.{suffix}"
        # if suffix_name == "":
        #     return f"{save_dir}/dp{dp_rank}_sep{sep_rank}_{param_name}.{suffix}"
        # else:
        #     return f"{save_dir}/dp{dp_rank}_sep{sep_rank}_{suffix_name}_{param_name}.{suffix}"

    def save(save_path, tensor):
        if save_as_numpy:
            np.save(save_path, tensor.numpy())
        else:
            assert isinstance(tensor, paddle.Tensor)
            paddle.save(tensor, save_path)

    if isinstance(input_data, dict):
        for k, v in input_data.items():
            save_path = save_path_name(dp_rank, sep_rank, k)
            assert isinstance(v, paddle.Tensor), f"{v} is not  paddle.Tensor"
            if save_once and os.path.exists(save_path):
                continue
            else:
                save(unique_name(save_path), v)
    elif isinstance(input_data, paddle.Tensor):
        save_path = save_path_name(dp_rank, sep_rank)
        if save_once and os.path.exists(save_path):
            return
        else:
            save(unique_name(save_path), input_data)
    else:
        raise ValueError(f"not support:{input_data}")

def split_inputs_sequence_dim(inputs, sep_rank=None, sep_degree=None):
    if sep_degree is None and sep_rank is None:
        _hcg = fleet.get_hybrid_communicate_group()
        sep_degree = _hcg.get_sep_parallel_world_size()
        sep_rank = _hcg.get_sep_parallel_rank()
    assert isinstance(sep_degree, int) and isinstance(sep_rank, int), f"sep_degree and sep_rank must be int"
    if sep_degree <= 1:
        return inputs
    def do_split_sequence_dim(data, sep_rank, split_sequence_len):
        if data is None:
            return None
        assert isinstance(data, paddle.Tensor), f"data should be paddle.Tensor, but is type:{type(data)}"
        assert len(data.shape) == 2, f"data dims should be 2, but shaped: {data.shape}"
        sliced_data = paddle.slice(
            data, [-1], [sep_rank * split_sequence_len], [(sep_rank + 1) * split_sequence_len])
        return sliced_data

    if isinstance(inputs, dict):
        res = {}
        for k, tensor in inputs.items():
            assert tensor.shape[-1] % sep_degree == 0, f"The last dim of {tensor} should be divisible by {sep_degree}"
            split_sequence_len = tensor.shape[-1] // sep_degree
            res[k] = do_split_sequence_dim(tensor, sep_rank, split_sequence_len)
    elif isinstance(inputs, list):
        res = []
        for tensor in inputs:
            res.append(do_split_sequence_dim(tensor))
    else:
        raise ValueError(f"the inputs should be a list or a dict, but is type: {type(inputs)}")
    return res


@paddle.no_grad()
def _reshard_qkv(x, group, split_axis=2, concat_axis=0):
    # [s/sep, b, h] -> [s, b, h/sep]
    # [s, b, h/sep] -> [s/sep, b, h]
    group = _get_global_group() if group is None else group
    nranks = dist.get_world_size(group=group)
    shape = x.shape

    assert len(shape) == 3, "Only support 3D tensor, but got {}".format(len(shape))
    assert shape[split_axis] % nranks == 0, "Only support evenly split, but got {} % {} != 0".format(shape[2], nranks)

    comm_tensor_list = paddle.split(x, nranks, axis=split_axis)
    output_list = [paddle.empty_like(comm_tensor_list[0]) for _ in comm_tensor_list]
    dist.alltoall(comm_tensor_list, output_list, group=group)
    reshard_tensor = paddle.concat(output_list, axis=concat_axis)

    return reshard_tensor


class ReshardQKV(PyLayer):
    @staticmethod
    def forward(ctx, x, group=None, split_axis=2, concat_axis=0):
        ctx.group = _get_global_group() if group is None else group
        ctx.split_axis = split_axis
        ctx.concat_axis = concat_axis
        res = _reshard_qkv(x, group, split_axis=ctx.split_axis, concat_axis=ctx.concat_axis)

        return res

    @staticmethod
    def backward(ctx, dy):
        res = _reshard_qkv(dy, ctx.group, split_axis=ctx.concat_axis, concat_axis=ctx.split_axis)

        return res


class ReshardLayer(paddle.nn.Layer):
    def __init__(self, sep_group=None) -> None:
        if sep_group is None:
            _hcg = fleet.get_hybrid_communicate_group()
            sep_group = _hcg.get_sep_parallel_group() if sep_group is None else sep_group
        self.sep_group = sep_group
        self.sep_degree = dist.get_world_size(group=self.sep_group)
        super(ReshardLayer, self).__init__()

    def forward(
        self,
        x,
        split_axis=1,
        concat_axis=2,
    ):
        # if x dims==3, its shape can be [s/sep, b, h] or [b, s/sep, h], the output shape can be [s, b, h/sep] or [b, s, h/sep]
        # if x dims==4, its shape can be [s, b, num_head/sep, head_dim] or [b, s, num_head/sep, head_dim], the output shape can be [s/sep, b, num_head, head_dim] or [b, s/sep, num_head, head_dim]
        shape = x.shape
        assert len(shape) == 3 or len(shape) == 4, "Only support 3D or 4D tensor"
        if len(shape) == 4:
            assert shape[split_axis] % self.sep_degree == 0
            shape[split_axis] = shape[split_axis] // self.sep_degree
            shape[concat_axis] = shape[concat_axis] * self.sep_degree

        input_data = x
        if len(shape) == 3:
            reshard_tensor = ReshardQKV.apply(input_data, self.sep_group, split_axis=split_axis, concat_axis=concat_axis)
        else:
            input_data = input_data.reshape([0, 0, -1])
            reshard_tensor = ReshardQKV.apply(input_data, self.sep_group, split_axis=split_axis, concat_axis=concat_axis)
            reshard_tensor.reshape_(shape)
        return reshard_tensor


class ReshardQKVUtest(PyLayer):
    """
    Reshard QKV for utest, it is only used for test.
    """

    @staticmethod
    def forward(ctx, x, group=None):
        # [s/sep, b, h] -> [s, b, h/sep]
        ctx.group = _get_global_group() if group is None else group
        ctx.nranks = dist.get_world_size(group=ctx.group)
        tensor_list = []
        paddle.distributed.all_gather(tensor_list, x, group=ctx.group)
        tensor = paddle.concat(tensor_list, axis=0)
        reshard_tensor = paddle.split(tensor, ctx.nranks, axis=2)[dist.get_rank(group=ctx.group)]
        return reshard_tensor

    @staticmethod
    def backward(ctx, dy):
        # [s, b, h / sep] - > [s / sep, b, h]
        tensor_list = []
        paddle.distributed.all_gather(tensor_list, dy, group=ctx.group)
        reshard_tensor = paddle.concat(tensor_list, axis=2)
        tensor = paddle.split(reshard_tensor, ctx.nranks, axis=0)[dist.get_rank(group=ctx.group)]
        return tensor


def test_qkv_out(x, y_grad, func, split_axis=2, concat_axis=0):
    x = x.detach()
    x.stop_gradient = False
    reshard_layer = ReshardLayer(sep_group=_get_global_group())
    y = reshard_layer(x, split_axis=split_axis, concat_axis=concat_axis)
    paddle.autograd.backward([y], [y_grad], True)
    return y, x.grad


def test_reshard_layer(x, y_grad, split_axis=0, concat_axis=2):
    x = x.detach()
    x.stop_gradient = False
    input_data = x
    reshard_layer = ReshardLayer(sep_group=_get_global_group())
    y = reshard_layer(
        x,
        split_axis=split_axis,
        concat_axis=concat_axis,
    )
    #
    paddle.autograd.backward([y], [y_grad], True)
    return y, x.grad


def prepare_manual_data():
    batch_size = 2
    seq_len = 2
    h = 4
    num_head = 2
    sep = dist.get_world_size()
    assert sep == 2, f"sep should be 2, but {sep}"
    num_elem = batch_size * seq_len // sep * num_head * h // num_head
    local_rank = dist.get_rank()
    input_data = paddle.to_tensor(
        np.reshape(
            np.arange(local_rank * num_elem, (local_rank + 1) * num_elem) + 1,
            [batch_size, seq_len, num_head // sep, h // num_head],
        ),
        dtype=paddle.float32,
    )
    if local_rank == 0:
        bin_bout_expected_output_data = paddle.to_tensor(
            np.reshape(np.array([1, 2, 9, 10, 5, 6, 13, 14]), [batch_size, seq_len // sep, num_head, h // num_head]),
            dtype=paddle.float32,
        )
        bin_sout_expected_output_data = paddle.to_tensor(
            np.reshape(np.array([1, 2, 9, 10, 5, 6, 13, 14]), [seq_len // sep, batch_size, num_head, h // num_head]),
            dtype=paddle.float32,
        )
        sin_bout_expected_output_data = paddle.to_tensor(
            np.reshape(np.array([1, 2, 9, 10, 3, 4, 11, 12]), [batch_size, seq_len // sep, num_head, h // num_head]),
            dtype=paddle.float32,
        )
        sin_sout_expected_output_data = paddle.to_tensor(
            np.reshape(np.array([1, 2, 9, 10, 3, 4, 11, 12]), [seq_len // sep, batch_size, num_head, h // num_head]),
            dtype=paddle.float32,
        )
    elif local_rank == 1:
        bin_bout_expected_output_data = paddle.to_tensor(
            np.reshape(np.array([3, 4, 11, 12, 7, 8, 15, 16]), [batch_size, seq_len // sep, num_head, h // num_head]),
            dtype=paddle.float32,
        )
        bin_sout_expected_output_data = paddle.to_tensor(
            np.reshape(np.array([3, 4, 11, 12, 7, 8, 15, 16]), [seq_len // sep, batch_size, num_head, h // num_head]),
            dtype=paddle.float32,
        )
        sin_bout_expected_output_data = paddle.to_tensor(
            np.reshape(np.array([5, 6, 13, 14, 7, 8, 15, 16]), [batch_size, seq_len // sep, num_head, h // num_head]),
            dtype=paddle.float32,
        )
        sin_sout_expected_output_data = paddle.to_tensor(
            np.reshape(np.array([5, 6, 13, 14, 7, 8, 15, 16]), [seq_len // sep, batch_size, num_head, h // num_head]),
            dtype=paddle.float32,
        )
    #
    input_data_list = []
    split_tensor_list = []
    for rank in range(sep):
        t = paddle.to_tensor(
            np.reshape(
                np.arange(rank * num_elem, (rank + 1) * num_elem) + 1,
                [batch_size, seq_len, num_head // sep, h // num_head],
            ),
            dtype=paddle.float32,
        )
        input_data_list.append(t)
        split_tensor_list.append(paddle.split(t, sep, axis=0))
    input_data = input_data_list[dist.get_rank()]
    expected_output = [s[local_rank] for s in split_tensor_list]
    expected_output = paddle.concat(expected_output, axis=2)
    np.testing.assert_equal(sin_sout_expected_output_data.numpy(), expected_output.numpy())
    return (
        input_data,
        bin_bout_expected_output_data,
        bin_sout_expected_output_data,
        sin_bout_expected_output_data,
        sin_sout_expected_output_data,
    )


def prepare_data(batch_major=True, dim_size=4, batch_size=2, seq_len=2, num_head=2, h=4, split_axis=0, concat_axis=2):
    assert dim_size == 3 or dim_size == 4, f"dim_size should be 3 or 4, but {dim_size}"
    assert split_axis == 0 or split_axis == 2
    batch_size = batch_size
    seq_len = seq_len
    h = h
    num_head = num_head
    sep = dist.get_world_size()
    assert sep == 2, f"sep should be 2, but {sep}"
    num_elem = batch_size * seq_len // sep * num_head * h // num_head
    local_rank = dist.get_rank()
    input_data_list = []
    split_tensor_list = []
    if dim_size == 4:
        shape = (
            [batch_size, seq_len, num_head // sep, h // num_head]
            if batch_major
            else [seq_len, batch_size, num_head // sep, h // num_head]
        )
        split_axis = 1 if batch_major else 0
        concat_axis = 2
    else:
        shape = [batch_size, seq_len // sep, h] if batch_major else [seq_len // sep, batch_size, h]
        split_axis = 2
        concat_axis = 1 if batch_major else 0
    for rank in range(sep):
        t = paddle.to_tensor(
            np.reshape(np.arange(rank * num_elem, (rank + 1) * num_elem) + 1, shape), dtype=paddle.float32
        )
        input_data_list.append(t)
        split_tensor_list.append(paddle.split(t, sep, axis=split_axis))
    input_data = input_data_list[dist.get_rank()]
    expected_output_data = [t[local_rank] for t in split_tensor_list]
    expected_output_data = paddle.concat(expected_output_data, axis=concat_axis)
    perm = [1, 0, 2, 3] if dim_size == 4 else [1, 0, 2]
    transpose_expected_output_data = paddle.transpose(expected_output_data, perm)
    print(
        f"input_data:{input_data}, expected_output_data:{expected_output_data}, transpose_expected_output_data:{transpose_expected_output_data}"
    )
    return input_data, expected_output_data, transpose_expected_output_data


def main():
    dist.init_parallel_env()
    test_split_inputs()
    test_reshard()

def test_split_inputs():
    batch_size = 8
    seq_len = 4096
    sep = dist.get_world_size()
    sep_rank = dist.get_rank()
    assert sep == 2, f"sep should be 2, but {sep}"
    
    inputs_ids = paddle.randint(low=0, high=65535, shape=(batch_size, seq_len))
    labels = paddle.randint(low=0, high=2, shape=(batch_size, seq_len))
    inputs = {"inputs_ids": inputs_ids, "labels": labels}

    splited_inputs = split_inputs_sequence_dim(inputs, sep_rank, sep)
    expected_local_inputs = {}
    for k, v in inputs.items():
        expected_local_inputs[k] = paddle.split(v, sep, axis=1)[sep_rank]
        assert k in splited_inputs
        np.testing.assert_equal(expected_local_inputs[k].numpy(), splited_inputs[k].numpy())


def test_reshard():
    # [s / sep, b, h] -> [s, b, h / sep]
    batch_size = 8
    seq_len = 16
    h = 12288
    sep = dist.get_world_size()
    assert seq_len % sep == 0, f"seq_len should be divisible by sep, seq_len:{seq_len}, sep:{sep}"

    # test reshard qkv
    x = paddle.randn([seq_len // sep, batch_size, h])
    y_grad = paddle.randn([seq_len, batch_size, h // sep])

    yb, xb_grad = test_qkv_out(x, y_grad, ReshardQKV.apply)
    y, x_grad = test_qkv_out(x, y_grad, ReshardQKVUtest.apply)

    np.testing.assert_equal(yb.numpy(), y.numpy())
    np.testing.assert_equal(xb_grad.numpy(), x_grad.numpy())

    def check_equal(input_data, expected_output_data, split_axis=0, concat_axis=2):
        bin_bout_output_grad = expected_output_data
        bin_bout_output, bin_bout_input_grad = test_reshard_layer(
            input_data,
            bin_bout_output_grad,
            split_axis=split_axis,
            concat_axis=concat_axis,
        )
        np.testing.assert_equal(bin_bout_output.numpy(), expected_output_data.numpy())
        np.testing.assert_equal(bin_bout_input_grad.numpy(), input_data.numpy())

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(
        batch_major=True, dim_size=3
    )
    check_equal(
        input_data,
        bin_bout_expected_output_data,
        split_axis=2,
        concat_axis=1,
    )
    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(
        batch_major=False, dim_size=3
    )
    check_equal(
        input_data,
        sin_sout_expected_output_data,
        split_axis=2,
        concat_axis=0,
    )

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(
        batch_major=True, dim_size=3, batch_size=2, seq_len=4, num_head=4, h=8
    )
    print(f"input_data shape: {input_data.shape}, bin_bout_expected_output_data shape: {bin_bout_expected_output_data.shape}")
    check_equal(
        input_data,
        bin_bout_expected_output_data,
        split_axis=2,
        concat_axis=1,
    )

    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(
        batch_major=False, dim_size=3, batch_size=2, seq_len=4, num_head=4, h=8
    )
    check_equal(
        input_data,
        sin_sout_expected_output_data,
        split_axis=2,
        concat_axis=0,
    )

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(
        batch_major=True, dim_size=3, batch_size=1, seq_len=4, num_head=4, h=8
    )
    check_equal(
        input_data,
        bin_bout_expected_output_data,
        split_axis=2,
        concat_axis=1,
    )

    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(
        batch_major=False, dim_size=3, batch_size=1, seq_len=4, num_head=4, h=8
    )
    check_equal(
        input_data,
        sin_sout_expected_output_data,
        split_axis=2,
        concat_axis=0,
    )
    print("testing reshard qkv pass")

    # test reshard out
    (
        input_data,
        bin_bout_expected_output_data,
        bin_sout_expected_output_data,
        sin_bout_expected_output_data,
        sin_sout_expected_output_data,
    ) = prepare_manual_data()

    input_data2, bin_bout_expected_output_data2, bin_sout_expected_output_data2 = prepare_data(batch_major=True)
    input_data2, sin_sout_expected_output_data2, sin_bout_expected_output_data2 = prepare_data(batch_major=False)

    def np_test(a, b):
        np.testing.assert_equal(a.numpy(), b.numpy())

    np_test(input_data, input_data2)
    np_test(bin_bout_expected_output_data, bin_bout_expected_output_data2)
    np_test(bin_sout_expected_output_data, bin_sout_expected_output_data2)
    np_test(sin_bout_expected_output_data, sin_bout_expected_output_data2)
    np_test(sin_sout_expected_output_data, sin_sout_expected_output_data2)

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(batch_major=True)
    check_equal(input_data, bin_bout_expected_output_data, split_axis=1, concat_axis=2)

    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(batch_major=False)
    check_equal(input_data, sin_sout_expected_output_data, split_axis=0, concat_axis=2)

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(
        batch_major=True, batch_size=2, seq_len=4, num_head=4, h=8
    )
    check_equal(input_data, bin_bout_expected_output_data, split_axis=1, concat_axis=2)

    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(
        batch_major=False, batch_size=2, seq_len=4, num_head=4, h=8
    )
    check_equal(input_data, sin_sout_expected_output_data, split_axis=0, concat_axis=2)

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(
        batch_major=True, batch_size=1, seq_len=4, num_head=4, h=8
    )
    check_equal(input_data, bin_bout_expected_output_data, split_axis=1, concat_axis=2)

    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(
        batch_major=False, batch_size=1, seq_len=4, num_head=4, h=8
    )
    check_equal(input_data, sin_sout_expected_output_data, split_axis=0, concat_axis=2)

    print("testing reshard output pass")


if __name__ == "__main__":
    main()
