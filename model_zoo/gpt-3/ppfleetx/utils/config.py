# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import argparse
import codecs
import copy
import os
import sys

import paddle
import paddle.distributed as dist
import yaml
from paddle.base.reader import use_pinned_memory

from . import check
from .log import advertise, logger

__all__ = ["get_config", "print_config"]


def process_dist_config(configs):
    """
    process distributed strategy for hybrid parallel
    """
    nranks = dist.get_world_size()

    config = configs["Distributed"]

    config.setdefault("hcg", "HybridCommunicateGroup")
    mp_degree = config.setdefault("mp_degree", 1)
    pp_degree = config.setdefault("pp_degree", 1)
    config.setdefault("pp_recompute_interval", 1)
    sep_degree = config.setdefault("sep_degree", 1)

    # sharding default
    sharding_config = config["sharding"]
    sharding_degree = sharding_config.setdefault("sharding_degree", 1)
    sharding_config.setdefault("sharding_stage", 2)
    sharding_config.setdefault("sharding_offload", False)
    reduce_overlap = sharding_config.setdefault("reduce_overlap", False)
    broadcast_overlap = sharding_config.setdefault("broadcast_overlap", False)

    other_degree = sep_degree * mp_degree * pp_degree * sharding_degree

    assert nranks % other_degree == 0, "unreasonable config of dist_strategy."
    dp_degree = config.setdefault("dp_degree", nranks // other_degree)
    assert nranks % dp_degree == 0, "unreasonable config of dist_strategy."
    assert nranks == dp_degree * other_degree, (
        "Mismatched config using {} cards with dp_degree[{}],"
        "sep_degree[{}], mp_degree[{}], pp_degree[{}] and sharding_degree[{}]".format(
            nranks, dp_degree, sep_degree, mp_degree, pp_degree, sharding_degree
        )
    )

    if sharding_config["sharding_degree"] > 1 and reduce_overlap:
        if sharding_config["sharding_stage"] == 3 or sharding_config["sharding_offload"]:
            sharding_config["reduce_overlap"] = False
            logger.warning("reduce overlap only valid for sharding stage 2 without offload")

    if sharding_config["sharding_degree"] > 1 and broadcast_overlap:
        if sharding_config["sharding_stage"] == 3 or sharding_config["sharding_offload"]:
            sharding_config["broadcast_overlap"] = False
            logger.warning("broadcast overlap only valid for sharding stage 2 without offload")

    if broadcast_overlap and configs["Engine"]["logging_freq"] == 1:
        logger.warning(
            "Set logging_freq to 1 will disable broadcast_overlap. "
            "If you want to overlap the broadcast, please increase the logging_freq."
        )
        sharding_config["broadcast_overlap"] = False

    if sharding_config["sharding_degree"] > 1:
        if getattr(sharding_config, "broadcast_overlap", False):
            logger.warning("Enable broadcast overlap for sharding will not use pin memory for dataloader")
            use_pinned_memory(False)

    if "fuse_sequence_parallel_allreduce" not in config:
        config["fuse_sequence_parallel_allreduce"] = False

    if "use_main_grad" in config and config["use_main_grad"] is True:
        logger.warning("If use_main_grad is True, fuse_sequence_parallel_allreduce will be forced to False")
        config["fuse_sequence_parallel_allreduce"] = False


def process_global_configs(config):
    """
    process global configs for hybrid parallel
    """
    dp_degree = config["Distributed"]["dp_degree"]
    pp_degree = config["Distributed"]["pp_degree"]
    sharding_degree = config["Distributed"]["sharding"]["sharding_degree"]

    config["Global"]["enable_partial_send_recv"] = config["Global"]["enable_partial_send_recv"] if "enable_partial_send_recv" in config["Global"] else True
    if "sequence_parallel" in config["Model"] and pp_degree > 1:
        if config["Model"]["sequence_parallel"]:
            config["Global"]["enable_partial_send_recv"] = False
            logger.warning(
                "if config.Distributed.pp_degree > 1 and config.Model.sequence_parallel is True, "
                "config.Global.enable_partial_send_recv will be set False."
            )

    global_cfg = config["Global"]

    # Set environment variable
    flags = global_cfg.get("flags", {})
    paddle.set_flags(flags)
    for k, v in flags.items():
        logger.info("Environment variable {} is set {}.".format(k, v))

    if global_cfg["global_batch_size"] is None and global_cfg["local_batch_size"] is None:
        raise ValueError("global_batch_size or local_batch_size should be set.")
    elif global_cfg["global_batch_size"] is not None and global_cfg["local_batch_size"] is not None:
        assert global_cfg["global_batch_size"] // global_cfg["local_batch_size"] == (dp_degree * sharding_degree), (
            "global_batch_size[{}] should be divided by local_batch_size[{}] "
            "when dp_degree is [{}] and sharding_degree is [{}]".format(
                global_cfg["global_batch_size"], global_cfg["local_batch_size"], dp_degree, sharding_degree
            )
        )
    elif global_cfg["global_batch_size"] is not None and global_cfg["local_batch_size"] is None:
        assert (
            global_cfg["global_batch_size"] % (dp_degree * sharding_degree) == 0
        ), "global_batch_size[{}] should be divided by dp_degree[{}] times sharding_degree[{}]".format(
            global_cfg["global_batch_size"], dp_degree, sharding_degree
        )
        global_cfg["local_batch_size"] = global_cfg["global_batch_size"] // (dp_degree * sharding_degree)
    else:
        global_cfg["global_batch_size"] = global_cfg["local_batch_size"] * dp_degree * sharding_degree
    assert global_cfg["local_batch_size"] % global_cfg["micro_batch_size"] == 0


def process_engine_config(config):
    """
    process engine
    """
    # save_load
    config.Engine["save_load"] = config.Engine.get("save_load", {})
    save_load_cfg = config.Engine.save_load
    save_steps = save_load_cfg.get("save_steps", None)
    save_epoch = save_load_cfg.get("save_epoch", None)
    if save_steps is None or save_steps == -1:
        save_load_cfg["save_steps"] = sys.maxsize if sys.version > "3" else sys.maxint

    if save_epoch is None or save_epoch == -1:
        save_load_cfg["save_epoch"] = 1

    save_load_cfg["output_dir"] = save_load_cfg.get("output_dir", "./output")
    save_load_cfg["ckpt_dir"] = save_load_cfg.get("ckpt_dir", None)

    # mix_precision
    config.Engine["mix_precision"] = config.Engine.get("mix_precision", {})
    amp_cfg = config.Engine.mix_precision

    amp_cfg["enable"] = amp_cfg.get("enable", False)
    amp_cfg["scale_loss"] = amp_cfg.get("scale_loss", 32768)
    amp_cfg["custom_black_list"] = amp_cfg.get("custom_black_list", None)
    amp_cfg["custom_white_list"] = amp_cfg.get("custom_white_list", None)

    # engine
    config.Engine["max_steps"] = config.Engine.get("max_steps", 500000)
    config.Engine["eval_freq"] = config.Engine.get("eval_freq", -1)
    config.Engine["eval_iters"] = config.Engine.get("eval_iters", 0)
    config.Engine["logging_freq"] = config.Engine.get("logging_freq", 1)
    config.Engine["num_train_epochs"] = config.Engine.get("num_train_epochs", 1)
    config.Engine["test_iters"] = (
        config.Engine["eval_iters"] * 10
        if config.Engine.get("test_iters", None) is None
        else config.Engine["test_iters"]
    )
    config.Engine["accumulate_steps"] = config.Global.local_batch_size // config.Global.micro_batch_size


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        for k, v in self.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def setdefault(self, k, default=None):
        if k not in self or self[k] is None:
            self[k] = default
            return default
        else:
            return self[k]


def create_attr_dict(yaml_config):
    from ast import literal_eval

    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""

    def _update_dic(dic, base_dic):
        """Update config from dic based base_dic"""
        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get("_inherited_", True) is False:
            dic.pop("_inherited_")
            return dic

        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = _update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(path):
        """Parse a yaml file and build config"""

        with codecs.open(path, "r", "utf-8") as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if "_base_" in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop("_base_")
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = _parse_from_yaml(base_path)
            dic = _update_dic(dic, base_dic)
        return dic

    yaml_dict = _parse_from_yaml(cfg_file)
    yaml_config = AttrDict(yaml_dict)

    create_attr_dict(yaml_config)
    return yaml_config


def print_dict(d, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    placeholder = "-" * 60
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", k))
            print_dict(v, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", k))
            for value in v:
                print_dict(value, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))
        if k.isupper():
            logger.info(placeholder)


def print_config(config):
    """
    visualize configs
    Arguments:
        config: configs
    """
    advertise()
    print_dict(config)


def check_config(config):
    """
    Check config
    """
    # global_batch_size = config.get("")

    global_config = config.get("Global")
    check.check_version()
    device = global_config.get("device", "gpu")
    device = device.lower()
    if device in ["gpu", "xpu", "rocm", "npu", "cpu"]:
        check.check_device(device)
    else:
        raise ValueError(
            f"device({device}) is not in ['gpu', 'xpu', 'rocm', 'npu', 'cpu'],\n"
            "Please ensure the config option Global.device is one of these devices"
        )


def override(dl, ks, v):
    """
    Recursively replace dict of list
    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
    """

    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    assert isinstance(dl, (list, dict)), "{} should be a list or a dict"
    assert len(ks) > 0, "lenght of keys should larger than 0"
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            assert k < len(dl), "index({}) out of range({})".format(k, dl)
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            # assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            if not ks[0] in dl:
                print(f"A new field ({ks[0]}) detected!")
            dl[ks[0]] = str2num(v)
        else:
            if ks[0] not in dl.keys():
                dl[ks[0]] = {}
                print(f"A new Series field ({ks[0]}) detected!")
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config
    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                'topk=2',
                'VALID.transforms.1.ResizeImage.resize_short=300'
            ]
    Returns:
        config(dict): replaced config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt, str), "option({}) should be a str".format(opt)
            assert "=" in opt, "option({}) should contain a =" "to distinguish between key and value".format(opt)
            pair = opt.split("=")
            assert len(pair) == 2, "there can be only a = in the option"
            key, value = pair
            keys = key.split(".")
            override(config, keys, value)
    return config


def get_config(fname, overrides=None, show=False):
    """
    Read config from file
    """
    assert os.path.exists(fname), "config file({}) is not exist".format(fname)
    config = parse_config(fname)
    override_config(config, overrides)

    process_dist_config(config)
    process_global_configs(config)
    process_engine_config(config)
    create_attr_dict(AttrDict(config))

    if show:
        print_config(config)
    check_config(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser("train script")
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml", help="config file path")
    parser.add_argument("-o", "--override", action="append", default=[], help="config options to be overridden")
    args = parser.parse_args()
    return args
