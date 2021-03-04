"""
Senta config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import json
import argparse

def str2bool(value):
    """
    String to Boolean
    """
    # because argparse does not support to parse "True, False" as python
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
        Init function for PDConfig.
        json_file: the path to the json configure file.
        """
        assert isinstance(json_file, str)

        self.args = None
        self.arg_config = {}

        parser = argparse.ArgumentParser()
        model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
        model_g.add_arg("ernie_config_path", str, None, "Path to the json file for ernie model config.")
        model_g.add_arg("senta_config_path", str, None, "Path to the json file for senta model config.")
        model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
        model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints")
        model_g.add_arg("model_type", str, "ernie_base", "Type of current ernie model")
        model_g.add_arg("use_paddle_hub", bool, False, "Whether to load ERNIE using PaddleHub")

        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
        train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
        train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
        train_g.add_arg("lr", float, 0.002, "The Learning rate value for training.")

        log_g = ArgumentGroup(parser, "logging", "logging related")
        log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
        log_g.add_arg("verbose", bool, False, "Whether to output verbose log")
        log_g.add_arg('enable_ce', bool, False, 'If set, run the task with continuous evaluation logs.')

        data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
        data_g.add_arg("data_dir", str, None, "Path to training data.")
        data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
        data_g.add_arg("batch_size", int, 256, "Total examples' number in batch for training.")
        data_g.add_arg("random_seed", int, 0, "Random seed.")
        data_g.add_arg("num_labels", int, 2, "label number")
        data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest sequence.")
        data_g.add_arg("train_set", str, None, "Path to training data.")
        data_g.add_arg("test_set", str, None, "Path to test data.")
        data_g.add_arg("dev_set", str, None, "Path to validation data.")
        data_g.add_arg("label_map_config", str, None, "label_map_path.")
        data_g.add_arg("do_lower_case", bool, True, "Whether to lower case the input text. Should be True for uncased models and False for cased models") 

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
        run_type_g.add_arg("task_name", str, None,
            "The name of task to perform sentiment classification.")
        run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
        run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation.")
        run_type_g.add_arg("do_infer", bool, True, "Whether to perform inference.")
        run_type_g.add_arg("do_save_inference_model", bool, True, "Whether to save inference model")
        run_type_g.add_arg("inference_model_dir", str, None, "Path to save inference model")
        
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
            with open(file_path, "r") as fin:
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
    pd_config = PDConfig('senta_config.json')
    pd_config.add_arg("my_age", int, 18, "I am forever 18.")
    pd_config.build()
    pd_config.print_arguments()
    print(pd_config.use_cuda)
    print(pd_config.model_type)
