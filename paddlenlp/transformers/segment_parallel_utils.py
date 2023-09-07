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


import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import _get_global_group
import numpy as np
import sys
sys.path.append("../../")
from paddlenlp.trainer.plugins import timer
from enum import Enum

# llama attention module in paddlenlp/transformers/llama/modeling.py-LlamaAttention
# gpt model_zoo/gpt-3/ppfleetx/models/language_model/gpt/dygraph/hybrid_model.py-MultiHeadAttention

class SplitAxis(Enum):
    SEQUENCE = 0
    HIDDEN = 2

@paddle.no_grad()
def _reshard_qkv(x, group, split_axis=2, concat_axis=0):
    # [s / sep, b, h] -> [s, b, h / sep]
    group = _get_global_group() if group is None else group
    nranks = dist.get_world_size(group=group)
    shape = x.shape

    assert len(shape) == 3, "Only support 3D tensor, but got {}".format(len(shape))
    assert shape[split_axis] % nranks == 0, "Only support evenly split, but got {} % {} != 0".format(shape[2], nranks)

    comm_tensor_list = paddle.split(x, nranks, axis=split_axis)
    output_list = [paddle.empty_like(comm_tensor_list[0]) for _ in comm_tensor_list]
    dist.alltoall(comm_tensor_list, output_list, group=group)
    reshard_tensor = paddle.concat(output_list, axis=concat_axis)
    # print(f"comm_tensor_list{comm_tensor_list}, output_list:{output_list}, reshard_tensor:{reshard_tensor}")

    return reshard_tensor


@paddle.no_grad()
def _reshard_output(x, group, split_axis=0, concat_axis=2, batch_major_in=True, batch_major_out=True):
    # batch_major_in=True, split_axis=0: [b, s, num_heads/sep, head_dim] --> [b, s/sep, num_heads, head_dim]
    # batch_major_in=True, split_axis=2: [b, s/sep, num_heads, head_dim] --> [b, s, num_heads/sep, head_dim]
    group = _get_global_group() if group is None else group
    nranks = dist.get_world_size(group=group)

    shape = x.shape
    assert len(shape) == 4, "Only support 4D tensor"

    # NOTE(shenliang03): if batch_size == 1, we don't need to transpose. 
    # It will be faster. Otherwise, we need to transpose batch_size behind seq_len.
    batch_size = shape[0]
    need_transpose = not (batch_size == 1)

    # transpose: [s/sep, b, h]
    x_tensor = x
    if batch_major_in:
        if need_transpose:
            x_tensor = paddle.transpose(x, [1, 0, 2, 3])
        else:
            x_tensor = x.reshape_([shape[1], shape[0], shape[2], shape[3]])

    assert x_tensor.shape[split_axis] % nranks == 0, f"Only support evenly split, x_tensor.shape:{x_tensor.shape}, split_axis:{split_axis} nranks:{nranks}"

    # convert to 3D tensor: [b, s/sep, h]
    x_tensor.reshape_([0, 0, shape[2] * shape[3]])

    # reshard_tensor = _to_head(x_tensor, group)
    reshard_tensor = _reshard_qkv(x_tensor, group, split_axis=split_axis, concat_axis=concat_axis)

    seq_dim_idx = 1 if batch_major_in else 0
    batch_dim_idx = 0 if batch_major_in else 1
    if split_axis == 0:
        # reshape: [s/sep, b, num_heads, head_dim]
        seq_dim = shape[seq_dim_idx] // nranks
        num_head_dim = shape[2] * nranks
        # reshard_tensor.reshape_([shape[1] // nranks, shape[0], shape[2] * nranks, shape[3]])
    elif split_axis == 2:
        # reshape: [s, b, num_heads/sep, head_dim]
        seq_dim = shape[seq_dim_idx] * nranks
        num_head_dim = shape[2] // nranks
        # reshard_tensor.reshape_([shape[1] * nranks, shape[0], shape[2] // nranks, shape[3]])
    else:
        raise ValueError(f"unknown split_axis:{split_axis}, should be 0 or 2")
    reshard_tensor.reshape_([seq_dim, shape[batch_dim_idx], num_head_dim, shape[3]])

    if batch_major_out:
        # transpose: [b, s, num_heads/sep, head_dim] or [b, s/sep, num_heads, head_dim]
        if need_transpose:
            reshard_tensor = paddle.transpose(reshard_tensor, [1, 0, 2, 3])
        else:
            reshard_tensor.reshape_([shape[0], seq_dim, 0, 0])
    print(f"reshard_tensor:{reshard_tensor}")
    return reshard_tensor

class ReshardLayer(paddle.nn.Layer):
    def __init__(self) -> None:
        super(ReshardLayer, self).__init__()

    def forward(self, x, group=None, split_axis=0, concat_axis=2, batch_major_in=False, batch_major_out=False):
        # x shape:[s/sep, b, h] or [s, b, num_head/sep, head_dim]
        shape = x.shape
        assert len(shape) == 3 or len(shape) == 4, "Only support 3D or 4D tensor"
        group = _get_global_group() if group is None else group
        nranks = dist.get_world_size(group=group)
        input_data = x
        perm = [1, 0, 2] if len(shape) == 3 else [1, 0, 2, 3]
        batch_dim_idx = 0 if batch_major_in else 1
        seq_dim_idx = 1 if batch_major_in else 0
        batch_size = shape[batch_dim_idx]
        seq_size = shape[seq_dim_idx]
        if batch_major_in:
            # NOTE(shenliang03): if batch_size == 1, we don't need to transpose. 
            # It will be faster. Otherwise, we need to transpose batch_size behind seq_len.
            if not (batch_size == 1):
                input_data = paddle.transpose(input_data, perm)
            else:
                new_shape = [seq_size, batch_size, 0] if len(shape) == 3 else [seq_size, batch_size, 0, 0]
                input_data = paddle.reshape(input_data, new_shape)

        if split_axis == 0:
            # reshape: [s/sep, b, num_heads, head_dim]
            resharded_seq_size = seq_size // nranks
            resharded_num_head_size = shape[2] * nranks
        elif split_axis == 2:
            # reshape: [s, b, num_heads/sep, head_dim]
            resharded_seq_size = seq_size * nranks
            resharded_num_head_size = shape[2] // nranks

        if len(shape) == 3:
            reshard_tensor = ReshardQKV.apply(input_data, group, split_axis=split_axis, concat_axis=concat_axis)
        else:
            input_data = paddle.reshape(input_data, [0, 0, -1])
            reshard_tensor = ReshardQKV.apply(input_data, group, split_axis=split_axis, concat_axis=concat_axis)
            
            
            reshard_tensor = paddle.reshape(reshard_tensor, [resharded_seq_size, batch_size, resharded_num_head_size, shape[3]])

        if batch_major_out:
            if not (batch_size == 1):
                reshard_tensor = paddle.transpose(reshard_tensor, perm)
            else:
                new_shape = [batch_size, resharded_seq_size, 0] if len(shape) == 3 else [batch_size, resharded_seq_size, resharded_num_head_size, shape[3]]
                reshard_tensor = paddle.reshape(reshard_tensor, new_shape)
        return reshard_tensor

@paddle.no_grad()
def _to_head(x, group):
    # [s / sep, b, h] -> [s, b, h / sep]
    group = _get_global_group() if group is None else group
    nranks = dist.get_world_size(group=group)
    shape = x.shape

    assert len(shape) == 3, "Only support 3D tensor, but got {}".format(len(shape))
    assert shape[2] % nranks == 0, "Only support evenly split, but got {} % {} != 0".format(shape[2], nranks)

    # [bs/sep, h/mp]
    x.reshape_([-1, shape[2]])

    comm_tensor = paddle.concat(paddle.split(x, nranks, axis=1),
                    axis=0)

    reshard_tensor = paddle.empty_like(comm_tensor)
    dist.alltoall(reshard_tensor, comm_tensor, group=group)
    reshard_tensor.reshape_([shape[0] * nranks, shape[1], shape[2] // nranks])

    return reshard_tensor

@paddle.no_grad()
def _to_seqlen(x, group):
    # [s, b, h / sep] -> [s / sep, b, h]
    group = _get_global_group() if group is None else group
    nranks = dist.get_world_size(group=group)
    shape = x.shape

    assert len(shape) == 3, "Only support 3D tensor, but got {}".format(len(shape))
    assert shape[0] % nranks == 0, "Only support evenly split, but got {} % {} != 0".format(shape[0], nranks)

    x.reshape_([-1, shape[2]])
    comm_tensor = paddle.empty_like(x)
    dist.alltoall(comm_tensor, x, group=group)

    reshard_tensor = paddle.concat(paddle.split(comm_tensor, nranks, axis=0),
                                   axis=1)
    reshard_tensor.reshape_([shape[0] // nranks, shape[1], shape[2] * nranks])
    return reshard_tensor

@paddle.no_grad()
def reshard_to_split_heads(x, group):
    # [b, s/sep, num_heads/mp, head_dim] --> [b, s, num_heads/(sep*mp), head_dim]
    group = _get_global_group() if group is None else group
    nranks = dist.get_world_size(group=group)

    shape = x.shape
    assert len(shape) == 4, "Only support 4D tensor"
    assert shape[2] % nranks == 0, "Only support evenly split"

    # NOTE(shenliang03): if batch_size == 1, we don't need to transpose. 
    # It will be faster. Otherwise, we need to transpose batch_size behind seq_len.
    batch_size = shape[0]
    need_transpose = not (batch_size == 1)

    # transpose: [s/sep, b, num_heads/mp, head_dim]
    x_tensor = paddle.transpose(x, [1, 0, 2, 3]) if need_transpose else x

    # convert to 3D tensor: [s/sep, b, h/mp]
    x_tensor.reshape_([0, 0, shape[2] * shape[3]])

    reshard_tensor = _to_head(x_tensor, group)

    # reshape: [s, b, num_heads/(sep*mp), head_dim]
    reshard_tensor.reshape_([shape[1] * nranks, shape[0], shape[2] // nranks, shape[3]])

    # transpose: [b, s, num_heads/(sep*mp), head_dim]
    if need_transpose:
        reshard_tensor = paddle.transpose(reshard_tensor, [1, 0, 2, 3])
    else:
        reshard_tensor.reshape_([shape[0], shape[1] * nranks, 0, 0])

    return reshard_tensor


@paddle.no_grad()
def reshard_to_split_seqlen(x, group):
    # [b, s, num_heads/sep, head_dim] ---> [b, s/sep, num_heads, head_dim] 
    group = _get_global_group() if group is None else group
    nranks = dist.get_world_size(group=group)
    shape = x.shape
    batch_size = shape[0]
    need_transpose = not (batch_size == 1)

    if need_transpose:
        x_tensor = paddle.transpose(x, [1, 0, 2, 3])
    else:
        x_tensor = x.reshape_([shape[1], shape[0], shape[2], shape[3]])

    # convert to 3D tensor
    x_tensor = x_tensor.reshape_([0, 0, shape[2] * shape[3]])

    reshard_tensor = _to_seqlen(x_tensor, group)
    
    # reshape: [s/sep, b, num_heads/mp, head_dim]
    reshard_tensor.reshape_([shape[1] // nranks, shape[0], shape[2] * nranks, shape[3]])
    if need_transpose:
        reshard_tensor = paddle.transpose(reshard_tensor, [1, 0, 2, 3])
    else:
        reshard_tensor.reshape_([shape[0], shape[1] // nranks, 0, 0])

    return reshard_tensor


class ReshardQKV(PyLayer):
    @staticmethod
    def forward(ctx, x, group=None, split_axis=2, concat_axis=0):
        _timer = timer.get_timers()
        _timer("reshard qkv fwd").start()
        ctx.group = _get_global_group() if group is None else group
        ctx.split_axis = split_axis
        ctx.concat_axis = concat_axis
        # with recovery_shape(x):
        # res = _to_head(x.clone(), group)
        res = _reshard_qkv(x.clone(), group, split_axis=ctx.split_axis, concat_axis=ctx.concat_axis)
        _timer("reshard qkv fwd").stop()

        return res

    @staticmethod
    def backward(ctx, dy):
        # with recovery_shape(dy):
        _timer = timer.get_timers()
        _timer("reshard qkv bwd").start()
        # res = _to_seqlen(dy, ctx.group)
        res = _reshard_qkv(dy, ctx.group, split_axis=ctx.concat_axis, concat_axis=ctx.split_axis)
        _timer("reshard qkv bwd").stop()

        return res


class ReshardOutProjection(PyLayer):
    @staticmethod
    def forward(ctx, x, group=None, split_axis=0, concat_axis=2, batch_major_in=True, batch_major_out=True):
        _timer = timer.get_timers()
        _timer("reshard out_proj fwd").start()
        # [b, s, num_heads/sep, head_dim] --> [b, s/sep, num_heads, head_dim]
        ctx.group = _get_global_group() if group is None else group
        ctx.split_axis = split_axis
        ctx.concat_axis = concat_axis
        ctx.batch_major_in = batch_major_in
        ctx.batch_major_out = batch_major_out
        res = _reshard_output(x.clone(), group, split_axis=split_axis, concat_axis=concat_axis, batch_major_in=batch_major_in, batch_major_out=batch_major_out)
        # res = reshard_to_split_seqlen(x.clone(), group)
        _timer("reshard out_proj fwd").stop()

        return res

    @staticmethod
    def backward(ctx, dy):
        # with recovery_shape(dy):
        _timer = timer.get_timers()
        _timer("reshard out_proj bwd").start()
        res = _reshard_output(dy, ctx.group, split_axis=ctx.concat_axis, concat_axis=ctx.split_axis, batch_major_in=ctx.batch_major_in, batch_major_out=ctx.batch_major_out)
        # res = reshard_to_split_heads(dy, ctx.group)
        _timer("reshard out_proj bwd").stop()

        return res


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
    # y = func(x)
    reshard_layer = ReshardLayer()
    y = reshard_layer(x, split_axis=split_axis, concat_axis=concat_axis)
    paddle.autograd.backward([y], [y_grad], True)
    return y, x.grad

def test_reshard_layer(x, y_grad, func, batch_major_in, batch_major_out, split_axis=0, concat_axis=2):
    x = x.detach()
    x.stop_gradient = False
    input_data = x
    reshard_layer = ReshardLayer()
    y = reshard_layer(x, split_axis=split_axis, concat_axis=concat_axis, batch_major_in=batch_major_in, batch_major_out=batch_major_out)
    #
    paddle.autograd.backward([y], [y_grad], True)
    print(f"x_ grad:{input_data.grad}, x.grad:{x.grad}")
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
        np.reshape(np.arange(local_rank * num_elem, (local_rank+1) * num_elem) + 1, [batch_size, seq_len, num_head // sep, h // num_head]), dtype=paddle.float32)
    if local_rank == 0:
        bin_bout_expected_output_data = paddle.to_tensor(np.reshape(np.array([1,2,9,10,5,6,13,14]), [batch_size, seq_len // sep, num_head, h // num_head]), dtype=paddle.float32)
        bin_sout_expected_output_data = paddle.to_tensor(np.reshape(np.array([1,2,9,10,5,6,13,14]), [seq_len // sep, batch_size, num_head, h // num_head]), dtype=paddle.float32)
        sin_bout_expected_output_data = paddle.to_tensor(np.reshape(np.array([1,2,9,10,3,4,11,12]), [batch_size, seq_len // sep, num_head, h // num_head]), dtype=paddle.float32)
        sin_sout_expected_output_data = paddle.to_tensor(np.reshape(np.array([1,2,9,10,3,4,11,12]), [seq_len // sep, batch_size, num_head, h // num_head]), dtype=paddle.float32)
    elif local_rank == 1:
        bin_bout_expected_output_data = paddle.to_tensor(np.reshape(np.array([3,4,11,12,7,8,15,16]), [batch_size, seq_len // sep, num_head, h // num_head]), dtype=paddle.float32)
        bin_sout_expected_output_data = paddle.to_tensor(np.reshape(np.array([3,4,11,12,7,8,15,16]), [seq_len // sep, batch_size, num_head, h // num_head]), dtype=paddle.float32)
        sin_bout_expected_output_data = paddle.to_tensor(np.reshape(np.array([5,6,13,14,7,8,15,16]), [batch_size, seq_len // sep, num_head, h // num_head]), dtype=paddle.float32)
        sin_sout_expected_output_data = paddle.to_tensor(np.reshape(np.array([5,6,13,14,7,8,15,16]), [seq_len // sep, batch_size, num_head, h // num_head]), dtype=paddle.float32)
    #
    input_data_list = []
    split_tensor_list = []
    for rank in range(sep):
        t = paddle.to_tensor(np.reshape(np.arange(rank * num_elem, (rank+1) * num_elem) + 1, [batch_size, seq_len, num_head // sep, h // num_head]), dtype=paddle.float32)
        input_data_list.append(t)
        split_tensor_list.append(paddle.split(t, sep, axis=0))
    input_data = input_data_list[dist.get_rank()]
    expected_output = [s[local_rank] for s in split_tensor_list]
    expected_output = paddle.concat(expected_output, axis=2)
    print(f"expected_output:{expected_output}")
    np.testing.assert_equal(sin_sout_expected_output_data.numpy(), expected_output.numpy())
    print(f"input_data:{input_data}, bin_bout_expected_output_data:{bin_bout_expected_output_data}, bin_sout_expected_output_data:{bin_sout_expected_output_data}, \
          sin_bout_expected_output_data:{sin_bout_expected_output_data}, sin_sout_expected_output_data:{sin_sout_expected_output_data}")
    return input_data, bin_bout_expected_output_data, bin_sout_expected_output_data, sin_bout_expected_output_data, sin_sout_expected_output_data

def prepare_data_auto(batch_size=2, seq_len=2, num_head=2, h=4):
    batch_size = batch_size
    seq_len = seq_len
    h = h
    num_head = num_head
    sep = dist.get_world_size()
    assert sep == 2, f"sep should be 2, but {sep}"
    num_elem = batch_size * seq_len // sep * num_head * h // num_head
    local_rank = dist.get_rank()
    input_data_list = []
    seq_major_split_tensor_list = []
    batch_major_split_tensor_list = []
    for rank in range(sep):
        t = paddle.to_tensor(np.reshape(np.arange(rank * num_elem, (rank+1) * num_elem) + 1, [seq_len, batch_size, num_head // sep, h // num_head]), dtype=paddle.float32)
        input_data_list.append(t)
        seq_major_split_tensor_list.append(paddle.split(t, sep, axis=0))
        batch_major_split_tensor_list.append(paddle.split(t, sep, axis=1))
    input_data = input_data_list[dist.get_rank()]
    sin_sout_expected_output_data = [s[local_rank] for s in seq_major_split_tensor_list]
    sin_sout_expected_output_data = paddle.concat(sin_sout_expected_output_data, axis=2)
    sin_bout_expected_output_data = paddle.transpose(sin_sout_expected_output_data, [1, 0, 2, 3])
    bin_bout_expected_output_data = [s[local_rank] for s in batch_major_split_tensor_list]
    bin_bout_expected_output_data = paddle.concat(bin_bout_expected_output_data, axis=2)
    bin_sout_expected_output_data = paddle.transpose(bin_bout_expected_output_data, [1, 0, 2, 3])
    print(f"prepare_data_auto, input_data:{input_data}, bin_bout_expected_output_data:{bin_bout_expected_output_data}, bin_sout_expected_output_data:{bin_sout_expected_output_data}, \
          sin_bout_expected_output_data:{sin_bout_expected_output_data}, sin_sout_expected_output_data:{sin_sout_expected_output_data}")
    return input_data, bin_bout_expected_output_data, bin_sout_expected_output_data, sin_bout_expected_output_data, sin_sout_expected_output_data

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
        shape = [batch_size, seq_len, num_head // sep, h // num_head] if batch_major else [seq_len, batch_size, num_head // sep, h // num_head]
        split_axis = 1 if batch_major else 0
        concat_axis = 2
    else:
        shape = [batch_size, seq_len // sep, h] if batch_major else [seq_len // sep, batch_size, h]
        split_axis = 2
        concat_axis = 1 if batch_major else 0
    for rank in range(sep):
        t = paddle.to_tensor(np.reshape(np.arange(rank * num_elem, (rank+1) * num_elem) + 1, shape), dtype=paddle.float32)
        input_data_list.append(t)
        split_tensor_list.append(paddle.split(t, sep, axis=split_axis))
    input_data = input_data_list[dist.get_rank()]
    expected_output_data = [t[local_rank] for t in split_tensor_list]
    expected_output_data = paddle.concat(expected_output_data, axis=concat_axis)
    perm = [1, 0, 2, 3] if dim_size == 4 else [1, 0, 2]
    transpose_expected_output_data = paddle.transpose(expected_output_data, perm)
    print(f"prepare_data_auto, input_data:{input_data}, expected_output_data:{expected_output_data}, transpose_expected_output_data:{transpose_expected_output_data}")
    return input_data, expected_output_data, transpose_expected_output_data

def main():
    dist.init_parallel_env()
    timer.set_timers()

    # [s / sep, b, h] -> [s, b, h / sep]
    batch_size = 8
    seq_len = 16
    h = 12288
    num_head = 64
    sep=dist.get_world_size()
    assert seq_len % sep == 0, f"seq_len should be divisible by sep, seq_len:{seq_len}, sep:{sep}"

    # test reshard qkv
    x = paddle.randn([seq_len // sep, batch_size, h])
    y_grad = paddle.randn([seq_len, batch_size, h // sep])

    yb, xb_grad = test_qkv_out(x, y_grad, ReshardQKV.apply)
    y, x_grad = test_qkv_out(x, y_grad, ReshardQKVUtest.apply)

    np.testing.assert_equal(yb.numpy(), y.numpy())
    np.testing.assert_equal(xb_grad.numpy(), x_grad.numpy())

    def test_pass(input_data, expected_output_data, batch_major_in, batch_major_out, split_axis=0, concat_axis=2):
        print(f"batch_major_in:{batch_major_in}, batch_major_out:{batch_major_out}")
        bin_bout_output_grad = expected_output_data
        bin_bout_output, bin_bout_input_grad = test_reshard_layer(input_data, bin_bout_output_grad, ReshardOutProjection.apply, batch_major_in=batch_major_in, batch_major_out=batch_major_out, split_axis=split_axis, concat_axis=concat_axis)
        np.testing.assert_equal(bin_bout_output.numpy(), expected_output_data.numpy())
        np.testing.assert_equal(bin_bout_input_grad.numpy(), input_data.numpy())
    
    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(batch_major=True, dim_size=3)
    test_pass(input_data, bin_bout_expected_output_data, batch_major_in=True, batch_major_out=True, split_axis=2, concat_axis=0)
    test_pass(input_data, bin_sout_expected_output_data, batch_major_in=True, batch_major_out=False, split_axis=2, concat_axis=0)
    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(batch_major=False, dim_size=3)
    test_pass(input_data, sin_sout_expected_output_data, batch_major_in=False, batch_major_out=False, split_axis=2, concat_axis=0)
    test_pass(input_data, sin_bout_expected_output_data, batch_major_in=False, batch_major_out=True, split_axis=2, concat_axis=0)

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(batch_major=True, dim_size=3, batch_size=2, seq_len=4, num_head=4, h=8)
    test_pass(input_data, bin_bout_expected_output_data, batch_major_in=True, batch_major_out=True, split_axis=2, concat_axis=0)
    test_pass(input_data, bin_sout_expected_output_data, batch_major_in=True, batch_major_out=False, split_axis=2, concat_axis=0)
    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(batch_major=False, dim_size=3, batch_size=2, seq_len=4, num_head=4, h=8)
    test_pass(input_data, sin_sout_expected_output_data, batch_major_in=False, batch_major_out=False, split_axis=2, concat_axis=0)
    test_pass(input_data, sin_bout_expected_output_data, batch_major_in=False, batch_major_out=True, split_axis=2, concat_axis=0)

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(batch_major=True, dim_size=3, batch_size=1, seq_len=4, num_head=4, h=8)
    test_pass(input_data, bin_bout_expected_output_data, batch_major_in=True, batch_major_out=True, split_axis=2, concat_axis=0)
    test_pass(input_data, bin_sout_expected_output_data, batch_major_in=True, batch_major_out=False, split_axis=2, concat_axis=0)
    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(batch_major=False, dim_size=3, batch_size=1, seq_len=4, num_head=4, h=8)
    test_pass(input_data, sin_sout_expected_output_data, batch_major_in=False, batch_major_out=False, split_axis=2, concat_axis=0)
    test_pass(input_data, sin_bout_expected_output_data, batch_major_in=False, batch_major_out=True, split_axis=2, concat_axis=0)
    print(f"testing reshard qkv pass")

    # test reshard out
    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data, sin_bout_expected_output_data, sin_sout_expected_output_data = prepare_manual_data()
    input_data2, bin_bout_expected_output_data2, bin_sout_expected_output_data2, sin_bout_expected_output_data2, sin_sout_expected_output_data2 = prepare_data_auto(batch_size=2)
    input_data2, bin_bout_expected_output_data2, bin_sout_expected_output_data2 = prepare_data(batch_major=True)
    input_data2, sin_sout_expected_output_data2, sin_bout_expected_output_data2  = prepare_data(batch_major=False)
    def np_test(a, b):
        np.testing.assert_equal(a.numpy(), b.numpy())
    np_test(input_data, input_data2)
    np_test(bin_bout_expected_output_data, bin_bout_expected_output_data2)
    np_test(bin_sout_expected_output_data, bin_sout_expected_output_data2)
    np_test(sin_bout_expected_output_data, sin_bout_expected_output_data2)
    np_test(sin_sout_expected_output_data, sin_sout_expected_output_data2)
    # return
    
    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(batch_major=True)
    test_pass(input_data, bin_bout_expected_output_data, batch_major_in=True, batch_major_out=True)
    test_pass(input_data, bin_sout_expected_output_data, batch_major_in=True, batch_major_out=False)
    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(batch_major=False)
    test_pass(input_data, sin_sout_expected_output_data, batch_major_in=False, batch_major_out=False)
    test_pass(input_data, sin_bout_expected_output_data, batch_major_in=False, batch_major_out=True)

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(batch_major=True, batch_size=2, seq_len=4, num_head=4, h=8)
    test_pass(input_data, bin_bout_expected_output_data, batch_major_in=True, batch_major_out=True)
    test_pass(input_data, bin_sout_expected_output_data, batch_major_in=True, batch_major_out=False)
    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(batch_major=False, batch_size=2, seq_len=4, num_head=4, h=8)
    test_pass(input_data, sin_sout_expected_output_data, batch_major_in=False, batch_major_out=False)
    test_pass(input_data, sin_bout_expected_output_data, batch_major_in=False, batch_major_out=True)

    input_data, bin_bout_expected_output_data, bin_sout_expected_output_data = prepare_data(batch_major=True, batch_size=1, seq_len=4, num_head=4, h=8)
    test_pass(input_data, bin_bout_expected_output_data, batch_major_in=True, batch_major_out=True)
    test_pass(input_data, bin_sout_expected_output_data, batch_major_in=True, batch_major_out=False)
    input_data, sin_sout_expected_output_data, sin_bout_expected_output_data = prepare_data(batch_major=False, batch_size=1, seq_len=4, num_head=4, h=8)
    test_pass(input_data, sin_sout_expected_output_data, batch_major_in=False, batch_major_out=False)
    test_pass(input_data, sin_bout_expected_output_data, batch_major_in=False, batch_major_out=True)

    print("testing reshard output pass")

if __name__ == "__main__":
     main()
