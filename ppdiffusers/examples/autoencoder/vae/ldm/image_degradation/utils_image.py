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

import math

import cv2
import numpy as np
import paddle


def narrow(x, axis, start, length):
    return paddle.slice(x, [axis], [start], [start + length])


def uint2single(img):
    return np.float32(img / 255.0)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.0).round())


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


# --------------------------------------------
# matlab's imwrite
# --------------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = paddle.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).astype(absx.dtype)) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * (((absx > 1) * (absx <= 2)).astype(absx.dtype))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = paddle.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = paddle.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.reshape([out_length, 1]).expand([out_length, P]) + paddle.linspace(0, P - 1, P).reshape(
        [1, P]
    ).expand([out_length, P])

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.reshape([out_length, 1]).expand([out_length, P]) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = paddle.sum(weights, 1).reshape([out_length, 1])
    weights = weights / weights_sum.expand([out_length, P])

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = paddle.sum((weights == 0).astype("int64"), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = narrow(indices, 1, 1, P - 2)
        weights = narrow(weights, 1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = narrow(indices, 1, 0, P - 2)
        weights = narrow(weights, 1, 0, P - 2)

    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# --------------------------------------------
# imresize for numpy image [0, 1]
# --------------------------------------------
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = paddle.to_tensor(img)
    need_squeeze = True if img.ndim == 2 else False
    if need_squeeze:
        img = img.unsqueeze(2)

    in_H, in_W, in_C = img.shape
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = "cubic"

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing
    )
    # process H dimension
    # symmetric copying
    img_aug = paddle.zeros([in_H + sym_len_Hs + sym_len_He, in_W, in_C])
    img_aug[sym_len_Hs : sym_len_Hs + in_H] = img

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = paddle.arange(sym_patch.shape[0] - 1, -1, -1).astype("int64")
    sym_patch_inv = sym_patch.index_select(inv_idx, axis=0)
    img_aug[:sym_len_Hs] = sym_patch_inv

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = paddle.arange(sym_patch.shape[0] - 1, -1, -1).astype("int64")
    sym_patch_inv = sym_patch.index_select(inv_idx, axis=0)
    img_aug[sym_len_Hs + in_H : sym_len_Hs + in_H + sym_len_He] = sym_patch_inv

    out_1 = paddle.zeros([out_H, in_W, in_C])
    kernel_width = weights_H.shape[1]
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx : idx + kernel_width, :, j].transpose([1, 0]).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = paddle.zeros([out_H, in_W + sym_len_Ws + sym_len_We, in_C])
    out_1_aug[:, sym_len_Ws : sym_len_Ws + in_W] = out_1

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = paddle.arange(sym_patch.shape[1] - 1, -1, -1).astype("int64")
    sym_patch_inv = sym_patch.index_select(inv_idx, axis=1)
    out_1_aug[:, :sym_len_Ws] = sym_patch_inv

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = paddle.arange(sym_patch.shape[1] - 1, -1, -1).astype("int64")
    sym_patch_inv = sym_patch.index_select(inv_idx, axis=1)
    out_1_aug[:, sym_len_Ws + in_W : sym_len_Ws + in_W + sym_len_We] = sym_patch_inv

    out_2 = paddle.zeros([out_H, out_W, in_C])
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx : idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2 = out_2.squeeze()

    return out_2.numpy()
