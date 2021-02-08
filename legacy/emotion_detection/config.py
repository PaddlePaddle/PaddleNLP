#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
EmoTect config
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import six
import json
import argparse

def str2bool(value):
    """
    String to Boolean
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return value.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    Argument Class
    """
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, dtype, default, help, **kwargs):
        """
        Add argument
        """
        dtype = str2bool if dtype == bool else dtype
        self._group.add_argument(
            "--" + name,
            default=default,
            type=dtype,
            help=help + ' Default: %(default)s.',
            **kwargs)


class PDConfig(object):
    """
    A high-level api for handling argument configs.
    """
    def __init__(self, json_file=""):
        """
        Init funciton for PDConfig.
        json_file: the path to the json configure file.
        """
        assert isinstance(json_file, str)

        self.args = None
        self.arg_config = {}

        parser = argparse.ArgumentParser()

        run_type_g = ArgumentGroup(parser, "Running type options", "")
        run_type_g.add_arg("do_train", bool, False, "Whether to perform training.")
        run_type_g.add_arg("do_val", bool, False, "Whether to perform evaluation.")
        run_type_g.add_arg("do_infer", bool, False, "Whether to perform inference.")
        run_type_g.add_arg("do_save_inference_model", bool, False, "Whether to perform save inference model.")

        model_g = ArgumentGroup(parser, "Model config options", "")
        model_g.add_arg("model_type", str, "cnn_net", "Model type to run the task.",
            choices=["bow_net","cnn_net", "lstm_net", "bilstm_net", "gru_net", "textcnn_net"])
        model_g.add_arg("num_labels", int, 3 , "Number of labels for classification")
        model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
        model_g.add_arg("save_checkpoint_dir", str, None, "Directory path to save checkpoints")
        model_g.add_arg("inference_model_dir", str, None, "Directory path to save inference model")


        data_g = ArgumentGroup(parser, "Data config options", "")
        data_g.add_arg("data_dir", str, None, "Directory path to training data.")
        data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
        data_g.add_arg("vocab_size", str, None, "Vocabulary size.")
        data_g.add_arg("max_seq_len", int, 128, "Number of words of the longest sequence.")

        train_g = ArgumentGroup(parser, "Training config options", "")
        train_g.add_arg("lr", float, 0.002, "The Learning rate value for training.")
        train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
        train_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")
        train_g.add_arg("batch_size", int, 256, "Total examples' number in batch for training.")
        train_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
        train_g.add_arg("save_steps", int, 1000, "The steps interval to save checkpoints.")
        train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
        train_g.add_arg("random_seed", int, 0, "Random seed.")

        log_g = ArgumentGroup(parser, "Logging options", "")
        log_g.add_arg("verbose", bool, False, "Whether to output verbose log")
        log_g.add_arg("task_name", str, None, "The name of task to perform emotion detection")
        log_g.add_arg('enable_ce', bool, False, 'If set, run the task with continuous evaluation logs.')

        custom_g = ArgumentGroup(parser, "Customize options", "")

        self.custom_g = custom_g
        self.parser = parser
        self.arglist = [a.dest for a in self.parser._actions]
        self.json_config = None

        if json_file != "":
            self.load_json(json_file)

    def load_json(self, file_path):
        """load json config """
        if not os.path.exists(file_path):
            raise Warning("the json file %s does not exist." % file_path)
            return

        try:
            with io.open(file_path, "r") as fin:
                self.json_config = json.load(fin)
        except Exception as e:
            raise IOError("Error in parsing json config file '%s'" % file_path)

        for name in self.json_config:
            # use `six.string_types` but not `str` for compatible with python2 and python3
            if not isinstance(self.json_config[name], (int, float, bool, six.string_types)):
                continue

            if name in self.arglist:
                self.set_default(name, self.json_config[name])
            else:
                self.custom_g.add_arg(name,
                                      type(self.json_config[name]),
                                      self.json_config[name],
                                      "customized options")

    def set_default(self, name, value):
        for arg in self.parser._actions:
            if arg.dest == name:
                arg.default = value

    def build(self):
        self.args = self.parser.parse_args()
        self.arg_config = vars(self.args)

    def print_arguments(self):
        print('-----------  Configuration Arguments -----------')
        for arg, value in sorted(six.iteritems(self.arg_config)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')

    def add_arg(self, name, dtype, default, descrip):
        self.custom_g.add_arg(name, dtype, default, descrip)

    def __add__(self, new_arg):
        assert isinstance(new_arg, list) or isinstance(new_arg, tuple)
        assert len(new_arg) >= 3
        assert self.args is None

        name = new_arg[0]
        dtype = new_arg[1]
        dvalue = new_arg[2]
        desc = new_arg[3] if len(new_arg) == 4 else "Description is not provided."

        self.add_arg(name, dtype, dvalue, desc)
        return self

    def __getattr__(self, name):
        if name in self.arg_config:
            return self.arg_config[name]

        if name in self.json_config:
            return self.json_config[name]

        raise Warning("The argument %s is not defined." % name)


if __name__ == '__main__':
    pd_config = PDConfig('config.json')
    pd_config += ("my_age", int, 18, "I am forever 18.")
    pd_config.build()
    pd_config.print_arguments()
    print(pd_config.use_cuda)
    print(pd_config.model_type)
