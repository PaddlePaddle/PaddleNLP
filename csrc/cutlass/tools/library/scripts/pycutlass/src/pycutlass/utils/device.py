#################################################################################################
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
#################################################################################################

"""
Utility functions for interacting with the device
"""

from cuda import cudart


def check_cuda_errors(result: list):
    """
    Checks whether `result` contains a CUDA error raises the error as an exception, if so. Otherwise,
    returns the result contained in the remaining fields of `result`.

    :param result: the results of the `cudart` method, consisting of an error code and any method results
    :type result: list

    :return: non-error-code results from the `results` parameter
    """
    # `result` is of the format : (cudaError_t, result...)
    err = result[0]
    if err.value:
        raise RuntimeError("CUDA error: {}".format(cudart.cudaGetErrorName(err)))

    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def device_cc(device: int = 0) -> int:
    """
    Returns the compute capability of the device with ID `device`.

    :param device: ID of the device to query
    :type device: int

    :return: compute capability of the queried device (e.g., 80 for SM80)
    :rtype: int
    """
    deviceProp = check_cuda_errors(cudart.cudaGetDeviceProperties(device))
    major = str(deviceProp.major)
    minor = str(deviceProp.minor)
    return int(major + minor)
