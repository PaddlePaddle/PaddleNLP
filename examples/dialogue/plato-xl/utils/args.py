#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Parse argument."""

import argparse
import json


def str2bool(v):
    """ Support bool type for argparse. """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


class Args(dict):
    """ Arguments class

    Store arguments in training / infer / ... scripts.
    """

    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        for v in self.values():
            if isinstance(v, Args):
                if name in v:
                    return v[name]
        return None

    def get(self, key, default_value=None):
        """Get the value of corresponding key."""
        if key in self.keys():
            return self[key]
        for v in self.values():
            if isinstance(v, Args):
                if key in v:
                    return v[key]
        return default_value

    def __setattr__(self, name, value):
        self[name] = value

    def save(self, filename):
        with open(filename, "w") as fp:
            json.dump(self, fp, ensure_ascii=False, indent=4, sort_keys=False)

    def load(self, filename, group_name=None):
        if group_name is not None:
            if group_name not in self:
                self[group_name] = Args()
            self[group_name].load(filename)
            return
        with open(filename, "r") as fp:
            params_dict = json.load(fp)
        for k, v in params_dict.items():
            if isinstance(v, dict):
                self[k].update(Args(v))
            else:
                self[k] = v


def parse_args(parser: argparse.ArgumentParser, allow_unknown=False) -> Args:
    """ Parse hyper-parameters from cmdline. """
    if allow_unknown:
        parsed, _ = parser.parse_known_args()
    else:
        parsed = parser.parse_args()
    args = Args()
    optional_args = parser._action_groups[1]
    for action in optional_args._group_actions[1:]:
        arg_name = action.dest
        args[arg_name] = getattr(parsed, arg_name)
    for group in parser._action_groups[2:]:
        group_args = Args()
        for action in group._group_actions:
            arg_name = action.dest
            group_args[arg_name] = getattr(parsed, arg_name)
        if len(group_args) > 0:
            if group.title in args:
                args[group.title].update(group_args)
            else:
                args[group.title] = group_args
    return args
