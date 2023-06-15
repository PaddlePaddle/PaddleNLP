# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import paddle

try:
    from paddle.framework import in_dygraph_mode
except ImportError:
    from paddle.fluid.framework import in_dygraph_mode


def _compute_quantile(x, q, axis=None, keepdim=False, ignore_nan=False):
    """
    Compute the quantile of the input along the specified axis.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64, int32, int64.
        q (int|float|list): The q for calculate quantile, which should be in range [0, 1]. If q is a list,
            each q will be calculated and the first dimension of output is same to the number of ``q`` .
        axis (int|list, optional): The axis along which to calculate quantile. ``axis`` should be int or list of int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is a list, quantile is calculated over all elements of given axises.
            If ``axis`` is None, quantile is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        ignore_nan: (bool, optional): Whether to ignore NaN of input Tensor.
            If ``ignore_nan`` is True, it will calculate nanquantile.
            Otherwise it will calculate quantile. Default is False.

    Returns:
        Tensor, results of quantile along ``axis`` of ``x``.
        In order to obtain higher precision, data type of results will be float64.
    """

    # Validate q
    if isinstance(q, (int, float)):
        q = [q]
    elif isinstance(q, (list, tuple)):
        if len(q) <= 0:
            raise ValueError("q should not be empty")
    else:
        raise TypeError("Type of q should be int, float, list or tuple.")

    # Validate axis
    dims = len(x.shape)
    out_shape = list(x.shape)
    if axis is None:
        x = paddle.flatten(x)
        axis = 0
        out_shape = [1] * dims
    else:
        if isinstance(axis, list):
            axis_src, axis_dst = [], []
            for axis_single in axis:
                if not isinstance(axis_single, int) or not (axis_single < dims and axis_single >= -dims):
                    raise ValueError(
                        "Axis should be None, int, or a list, element should in range [-rank(x), rank(x))."
                    )
                if axis_single < 0:
                    axis_single = axis_single + dims
                axis_src.append(axis_single)
                out_shape[axis_single] = 1

            axis_dst = list(range(-len(axis), 0))
            x = paddle.moveaxis(x, axis_src, axis_dst)
            if len(axis_dst) == 0:
                x = paddle.flatten(x)
                axis = 0
            else:
                x = paddle.flatten(x, axis_dst[0], axis_dst[-1])
                axis = axis_dst[0]
        else:
            if not isinstance(axis, int) or not (axis < dims and axis >= -dims):
                raise ValueError("Axis should be None, int, or a list, element should in range [-rank(x), rank(x)).")
            if axis < 0:
                axis += dims
            out_shape[axis] = 1

    mask = x.isnan()
    valid_counts = mask.logical_not().sum(axis=axis, keepdim=True, dtype="float64")

    indices = []

    for q_num in q:
        if q_num < 0 or q_num > 1:
            raise ValueError("q should be in range [0, 1]")
        if in_dygraph_mode():
            q_num = paddle.to_tensor(q_num, dtype="float64")
        if ignore_nan:
            indices.append(q_num * (valid_counts - 1))
        else:
            # TODO: Use paddle.index_fill instead of where
            index = q_num * (valid_counts - 1)
            last_index = x.shape[axis] - 1
            nums = paddle.full_like(index, fill_value=last_index)
            index = paddle.where(mask.any(axis=axis, keepdim=True), nums, index)
            indices.append(index)

    # sorted_tensor = paddle.sort(x, axis)
    data, _ = paddle.topk(x, k=x.shape[axis], axis=axis, largest=False)
    sorted_tensor = data

    outputs = []

    # TODO(chenjianye): replace the for-loop to directly take elements.
    for index in indices:
        indices_below = paddle.floor(index).astype(paddle.int32)
        indices_upper = paddle.ceil(index).astype(paddle.int32)
        tensor_upper = paddle.take_along_axis(sorted_tensor, indices_upper, axis=axis)
        tensor_below = paddle.take_along_axis(sorted_tensor, indices_below, axis=axis)
        weights = index - indices_below.astype("float64")
        out = paddle.lerp(
            tensor_below.astype("float64"),
            tensor_upper.astype("float64"),
            weights,
        )
        if not keepdim:
            out = paddle.squeeze(out, axis=axis)
        else:
            out = out.reshape(out_shape)
        outputs.append(out)

    if len(q) > 1:
        outputs = paddle.stack(outputs, 0)
    else:
        outputs = outputs[0]

    return outputs


def quantile(x, q, axis=None, keepdim=False):
    """
    Compute the quantile of the input along the specified axis.
    If any values in a reduced row are NaN, then the quantiles for that reduction will be NaN.

    Args:
        x (Tensor): The input Tensor, it's data type can be float32, float64, int32, int64.
        q (int|float|list): The q for calculate quantile, which should be in range [0, 1]. If q is a list,
            each q will be calculated and the first dimension of output is same to the number of ``q`` .
        axis (int|list, optional): The axis along which to calculate quantile. ``axis`` should be int or list of int.
            ``axis`` should be in range [-D, D), where D is the dimensions of ``x`` .
            If ``axis`` is less than 0, it works the same way as :math:`axis + D`.
            If ``axis`` is a list, quantile is calculated over all elements of given axises.
            If ``axis`` is None, quantile is calculated over all elements of ``x``. Default is None.
        keepdim (bool, optional): Whether to reserve the reduced dimension(s)
            in the output Tensor. If ``keepdim`` is True, the dimensions of
            the output Tensor is the same as ``x`` except in the reduced
            dimensions(it is of size 1 in this case). Otherwise, the shape of
            the output Tensor is squeezed in ``axis`` . Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, results of quantile along ``axis`` of ``x``.
        In order to obtain higher precision, data type of results will be float64.

    Examples:
        .. code-block:: python

            import paddle

            y = paddle.arange(0, 8 ,dtype="float32").reshape([4, 2])
            # Tensor(shape=[4, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            #        [[0., 1.],
            #         [2., 3.],
            #         [4., 5.],
            #         [6., 7.]])

            y1 = paddle.quantile(y, q=0.5, axis=[0, 1])
            # Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        3.50000000)

            y2 = paddle.quantile(y, q=0.5, axis=1)
            # Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [0.50000000, 2.50000000, 4.50000000, 6.50000000])

            y3 = paddle.quantile(y, q=[0.3, 0.5], axis=0)
            # Tensor(shape=[2, 2], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [[1.80000000, 2.80000000],
            #         [3.        , 4.        ]])

            y[0,0] = float("nan")
            y4 = paddle.quantile(y, q=0.8, axis=1, keepdim=True)
            # Tensor(shape=[4, 1], dtype=float64, place=Place(cpu), stop_gradient=True,
            #        [[nan       ],
            #         [2.80000000],
            #         [4.80000000],
            #         [6.80000000]])

    """
    return _compute_quantile(x, q, axis=axis, keepdim=keepdim, ignore_nan=False)
