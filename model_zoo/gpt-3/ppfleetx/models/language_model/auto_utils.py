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

from functools import reduce

import numpy as np
import paddle.distributed.auto_parallel as auto


def process_mesh_config(config):
    class Mesh:
        def __init__(self, config):
            self.dp_dim = None
            self.mp_dim = None
            self.process_mesh = None
            self.config = config

            topology = list(
                filter(lambda x: x > 1, [self.config["pp_degree"], self.config["dp_degree"], self.config["mp_degree"]])
            )
            num_proc = 1 if not topology else reduce(lambda x, y: x * y, topology)
            processes = [i for i in range(num_proc)]

            if self.config["pp_degree"] > 1:
                if len(topology) > 1:
                    # dpmppp, dppp, mppp
                    if len(topology) > 2:
                        # dpmppp
                        self.process_mesh = auto.ProcessMesh(
                            np.array(processes).reshape(topology), dim_names=["pp", "dp", "mp"]
                        )
                        self.dp_dim = "dp"
                        self.mp_dim = "mp"
                    elif self.config["dp_degree"] > 1:
                        # dppp
                        self.process_mesh = auto.ProcessMesh(
                            np.array(processes).reshape(topology), dim_names=["pp", "dp"]
                        )
                        self.dp_dim = "dp"
                    elif self.config["mp_degree"] > 1:
                        # mppp
                        self.process_mesh = auto.ProcessMesh(
                            np.array(processes).reshape(topology), dim_names=["pp", "mp"]
                        )
                        self.mp_dim = "mp"
                elif len(topology) == 1:
                    # pp
                    self.process_mesh = auto.ProcessMesh(processes, dim_names=["pp"])
            else:
                if len(topology) > 1:
                    # dpmp
                    self.process_mesh = auto.ProcessMesh(np.array(processes).reshape(topology), dim_names=["dp", "mp"])
                    self.dp_dim = "dp"
                    self.mp_dim = "mp"
                elif self.config["dp_degree"] > 1:
                    # dp
                    self.process_mesh = auto.ProcessMesh(processes, dim_names=["dp"])
                    self.dp_dim = "dp"
                elif self.config["mp_degree"] > 1:
                    # mp
                    self.process_mesh = auto.ProcessMesh(processes, dim_names=["mp"])
                    self.mp_dim = "mp"
                else:
                    # serial
                    self.process_mesh = auto.ProcessMesh(processes)

        def __getitem__(self, idx):

            if "pp" in self.process_mesh.dim_names:
                return self.process_mesh[idx]

            return self.process_mesh

        def stages(self, num_layers):
            layer_per_stage = num_layers // self.config["pp_degree"]
            return [i // layer_per_stage for i in range(num_layers)]

        @property
        def dp(self):
            return self.dp_dim

        @property
        def mp(self):
            return self.mp_dim

    return Mesh(config)


def process_model_configs(config):
    """
    process model configs for auto parallel
    """
    cfg_model = config["Model"]
    mesh = process_mesh_config(config["Distributed"])
    cfg_model.update({"mesh": mesh})
    if cfg_model["ffn_hidden_size"] is None:
        cfg_model["ffn_hidden_size"] = 4 * cfg_model["hidden_size"]

    if cfg_model["use_recompute"]:
        if not cfg_model.get("recompute_granularity", None):
            cfg_model["recompute_granularity"] = "full"


def process_data_configs(config):
    """
    process data configs for auto parallel
    """
    cfg_global = config["Global"]
    cfg_data = config["Data"]

    mode_to_num_samples = {
        "Train": cfg_global["global_batch_size"] * config["Engine"]["max_steps"],
        "Eval": cfg_global["global_batch_size"]
        * (config["Engine"]["max_steps"] // config["Engine"]["eval_freq"] + 1)
        * config["Engine"]["eval_iters"],
        "Test": cfg_global["global_batch_size"] * config["Engine"]["test_iters"],
    }

    for mode in ("Train", "Eval", "Test"):
        if mode in cfg_data.keys():
            cfg_data[mode]["dataset"]["num_samples"] = mode_to_num_samples[mode]
            cfg_data[mode]["dataset"]["mode"] = mode
            cfg_data[mode]["dataset"]["seed"] = cfg_global["seed"]


def process_configs(config):

    process_model_configs(config)
    process_data_configs(config)

    return config
