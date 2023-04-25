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


import math

import cv2
import numpy as np
import paddle

"""
M-LSD
Copyright 2021-present NAVER Corp.
Apache License v2.0
"""


def normal_(tensor, mean=0.0, std=1.0):
    """
    Modified tensor inspace using normal_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        mean (float|int): mean value.
        std (float|int): std value.
    Return:
        tensor
    """
    return _no_grad_normal_(tensor, mean, std)


def zeros_(tensor):
    """
    Modified tensor inspace using zeros_
    Args:
        tensor (paddle.Tensor): paddle Tensor
    Return:
        tensor
    """
    return _no_grad_fill_(tensor, 0)


def kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", reverse=False):
    """
    Modified tensor inspace using kaiming_normal_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        mode (str): ['fan_in', 'fan_out'], 'fin_in' defalut
        nonlinearity (str): nonlinearity method name
        reverse (bool):  reverse (bool: False): tensor data format order, False by default as [fout, fin, ...].
    Return:
        tensor
    """
    fan = _calculate_correct_fan(tensor, mode, reverse)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return _no_grad_normal_(tensor, 0, std)


def _no_grad_fill_(tensor, value=0.0):
    with paddle.no_grad():
        tensor.set_value(paddle.full_like(tensor, value, dtype=tensor.dtype))
    return tensor


def _no_grad_normal_(tensor, mean=0.0, std=1.0):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean=mean, std=std, shape=tensor.shape))
    return tensor


def _calculate_gain(nonlinearity, param=None):
    linear_fns = ["linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


# reference: https://pytorch.org/docs/stable/_modules/torch/nn/init.html
def _calculate_correct_fan(tensor, mode, reverse=False):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, reverse)

    return fan_in if mode == "fan_in" else fan_out


def _calculate_fan_in_and_fan_out(tensor, reverse=False):
    """
    Calculate (fan_in, _fan_out) for tensor
    Args:
        tensor (Tensor): paddle.Tensor
        reverse (bool: False): tensor data format order, False by default as [fout, fin, ...]. e.g. : conv.weight [cout, cin, kh, kw] is False; linear.weight [cin, cout] is True
    Return:
        Tuple[fan_in, fan_out]
    """
    if tensor.ndim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if reverse:
        num_input_fmaps, num_output_fmaps = tensor.shape[0], tensor.shape[1]
    else:
        num_input_fmaps, num_output_fmaps = tensor.shape[1], tensor.shape[0]

    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = np.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def deccode_output_score_and_ptss(tpMap, topk_n=200, ksize=5):
    """
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    """
    b, c, h, w = tpMap.shape
    assert b == 1, "only support bsize==1"
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, (0), :, :]
    heat = paddle.nn.functional.sigmoid(x=center).unsqueeze(0)
    hmax = paddle.nn.functional.max_pool2d(
        kernel_size=(ksize, ksize), stride=1, padding=(ksize - 1) // 2, x=heat
    ).squeeze(0)
    keep = (hmax == heat).astype(dtype="float32")
    heat = heat * keep
    heat = heat.reshape([-1])
    scores, indices = paddle.topk(x=heat, k=topk_n, axis=-1, largest=True)
    w_t = paddle.to_tensor(w)
    yy = paddle.floor_divide(x=indices, y=w_t).unsqueeze(axis=-1)
    xx = paddle.mod(indices, w_t).unsqueeze(axis=-1)
    ptss = paddle.concat(x=(yy, xx), axis=-1)
    ptss = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()
    displacement = displacement.transpose((1, 2, 0))
    return ptss, scores, displacement


def pred_lines(image, model, input_shape=[512, 512], score_thr=0.1, dist_thr=20.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]
    resized_image = np.concatenate(
        [
            cv2.resize(image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA),
            np.ones([input_shape[0], input_shape[1], 1]),
        ],
        axis=-1,
    )
    resized_image = resized_image.transpose((2, 0, 1))
    batch_image = np.expand_dims(resized_image, axis=0).astype("float32")
    batch_image = batch_image / 127.5 - 1.0
    batch_image = paddle.to_tensor(data=batch_image).astype(dtype="float32")
    outputs = model(batch_image)
    pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))
    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[(y), (x), :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
    lines = 2 * np.array(segments_list)
    lines[:, (0)] = lines[:, (0)] * w_ratio
    lines[:, (1)] = lines[:, (1)] * h_ratio
    lines[:, (2)] = lines[:, (2)] * w_ratio
    lines[:, (3)] = lines[:, (3)] * h_ratio
    return lines
