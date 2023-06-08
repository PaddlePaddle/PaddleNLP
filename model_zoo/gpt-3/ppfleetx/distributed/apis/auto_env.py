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

import os

import numpy as np
import paddle
import paddle.distributed.auto_parallel as auto

_mesh = None


def get_mesh():
    global _mesh
    if _mesh is None and paddle.distributed.get_world_size() == 1:
        set_mesh(
            Mesh(
                get_local_rank(),
                1,
                1,
                1,
            )
        )
    return _mesh


def set_mesh(mesh):
    global _mesh
    _mesh = mesh


class Mesh:
    def __init__(self, rank, dp_degree, mp_degree, pp_degree):
        self._dp_dim = "dp" if dp_degree > 1 else None
        self._mp_dim = "mp" if mp_degree > 1 else None
        self._dp_degree = dp_degree
        self._mp_degree = mp_degree
        self._pp_degree = pp_degree

        arr = np.arange(0, pp_degree * dp_degree * mp_degree).reshape([dp_degree, pp_degree, mp_degree])
        arr = arr.transpose(1, 0, 2)
        self.world_process_mesh = auto.ProcessMesh(arr, dim_names=["pp", "dp", "mp"])
        self.g_process_mesh = auto.ProcessMesh(list(range(pp_degree * dp_degree * mp_degree)))
        ipp, idp, imp = np.where(arr == rank)
        ipp = ipp[0]
        idp = idp[0]
        imp = imp[0]

        if dp_degree > 1 and mp_degree > 1:
            self.pp_process_mesh = self.world_process_mesh
        elif mp_degree > 1:
            self.pp_process_mesh = self.world_process_mesh[:, idp, :]
        else:
            self.pp_process_mesh = self.world_process_mesh[:, :, imp]

    @property
    def dp_degree(self):
        return self._dp_degree

    @property
    def mp_degree(self):
        return self._mp_degree

    @property
    def pp_degree(self):
        return self._pp_degree

    @property
    def dp_dim(self):
        return self._dp_dim

    @property
    def mp_dim(self):
        return self._mp_dim

    def __getitem__(self, idx):
        return self.pp_process_mesh[idx]


def init_dist_env(config):
    paddle.set_device(config.Global.device)

    mesh = Mesh(
        get_local_rank(),
        config.Distributed.dp_degree,
        config.Distributed.mp_degree,
        config.Distributed.pp_degree,
    )
    set_mesh(mesh)
    paddle.distributed.fleet.init(is_collective=True)


def get_local_rank():
    return int(os.getenv("PADDLE_RANK_IN_NODE", 0))
