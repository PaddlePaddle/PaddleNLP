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


def process_optim_configs(config):
    """
    process optim configs for auto parallel
    """
    config["Optimizer"]["lr"]["decay_steps"] *= config["Global"]["global_batch_size"]


def process_model_configs(config):
    """
    process model configs for auto parallel
    """
    cfg_model = config["Model"]
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
    process_optim_configs(config)

    return config
