#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import traceback
import logging
from types import SimpleNamespace
import argparse
import json
import _jsonnet as jsonnet


def define_args_parser():
    """define command-line args parser"""

    def _arg_bool(arg):
        """trans arg to bool type
        """
        if arg is None:
            return arg

        if type(arg) is not str:
            return bool(arg)

        if arg.isdigit():
            return bool(int(arg))

        if arg.lower() == 'true':
            return True
        else:
            return False

    parser = argparse.ArgumentParser(
        description="text2sql command-line interface")
    parser.add_argument(
        '-c',
        '--config',
        default='conf/text2sql.jsonnet',
        help='global config file path. it\'s priority is the lowest')

    general_args = parser.add_argument_group(title='general')
    general_args.add_argument(
        '--mode',
        type=str.lower,
        default='debug',
        required=False,
        choices=['preproc', 'train', 'infer', 'test', 'debug'])
    general_args.add_argument('--batch-size', type=int)
    general_args.add_argument('--beam-size', default=1, type=int)
    general_args.add_argument("--use-cuda",
                              type=_arg_bool,
                              default=True,
                              help="is run in cuda mode")
    general_args.add_argument("--is-eval-value",
                              type=_arg_bool,
                              default=True,
                              help="is evaluating value")
    general_args.add_argument("--is-cloud",
                              type=_arg_bool,
                              default=False,
                              help="is run in paddle cloud")
    general_args.add_argument("--is-debug",
                              type=_arg_bool,
                              default=False,
                              help="is run in debug mode")

    model_args = parser.add_argument_group(title='model')
    model_args.add_argument('--pretrain-model',
                            help='ernie model path for dygraph')
    model_args.add_argument('--init-model-params', help='trained model params')
    model_args.add_argument('--init-model-optim', help='dumped model optimizer')
    model_args.add_argument('--model-name',
                            choices=['seq2tree_v2'],
                            help='ernie model path for dygraph')
    model_args.add_argument('--grammar-type',
                            choices=['dusql_v2', 'nl2sql'],
                            help='')

    data_args = parser.add_argument_group(title='data')
    data_args.add_argument('--data-root', help='root data path. low priority.')
    data_args.add_argument(
        '--db',
        help='a tuple of pathes (schema, content) or path to dumped file')
    data_args.add_argument('--db-schema', help='temp argument')
    data_args.add_argument(
        '--grammar',
        help='path to grammar definition file, or cached label vocabs directory'
    )
    data_args.add_argument('--train-set',
                           help='original dataset path or dumped file path')
    data_args.add_argument('--dev-set',
                           help='original dataset path or dumped file path')
    data_args.add_argument('--test-set',
                           help='original dataset path or dumped file path')
    data_args.add_argument('--eval-file',
                           help='file to be evaluated(inferenced result)')
    data_args.add_argument('--output', help='')
    data_args.add_argument("--is-cached",
                           type=_arg_bool,
                           help="is dataset in cached format")

    train_args = parser.add_argument_group(title='train')
    train_args.add_argument('--epochs', type=int)
    train_args.add_argument('--learning-rate', type=float)
    train_args.add_argument('--log-steps', type=int)
    train_args.add_argument('--random-seed', type=int)
    train_args.add_argument('--use-data-parallel', type=_arg_bool)

    return parser


def gen_config(arg_list=None):
    """read configs from file, and updating it by command-line arguments

    Args:
        config_path (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    parser = define_args_parser()
    cli_args = parser.parse_args(arg_list)
    if cli_args.data_root is not None:
        root_path = cli_args.data_root
        if cli_args.is_cached or cli_args.is_cached is None:
            if cli_args.db is None:
                cli_args.db = os.path.join(root_path, 'db.pkl')
            if cli_args.grammar is None:
                cli_args.grammar = os.path.join(root_path, 'label_vocabs')
            if cli_args.train_set is None:
                cli_args.train_set = os.path.join(root_path, 'train.pkl')
            if cli_args.dev_set is None:
                cli_args.dev_set = os.path.join(root_path, 'dev.pkl')
            if cli_args.test_set is None and not cli_args.mode.startswith(
                    'train'):
                cli_args.test_set = os.path.join(root_path, 'test.pkl')
        else:
            if cli_args.db is None:
                cli_args.db = [
                    os.path.join(root_path, 'db_schema.json'),
                    os.path.join(root_path, 'db_content.json')
                ]
            if cli_args.train_set is None:
                cli_args.train_set = os.path.join(root_path, 'train.json')
            if cli_args.dev_set is None:
                cli_args.dev_set = os.path.join(root_path, 'dev.json')
            if cli_args.test_set is None and not cli_args.mode.startswith(
                    'train'):
                cli_args.test_set = os.path.join(root_path, 'test.json')

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {
            arg.dest: getattr(cli_args, arg.dest, None)
            for arg in group._group_actions
        }
        arg_groups[group.title] = {
            k: v
            for k, v in group_dict.items() if v is not None
        }

    config_file = cli_args.config
    config = json.loads(jsonnet.evaluate_file(config_file),
                        object_hook=lambda o: SimpleNamespace(**o))

    for group, args in arg_groups.items():
        if not hasattr(config, group):
            logging.debug(f'group {group} is not a module of config')
            setattr(config, group, SimpleNamespace())
        config_module = getattr(config, group)
        for name, value in args.items():
            setattr(config_module, name, value)

    return config


if __name__ == "__main__":
    """run some simple test cases"""
    print(gen_config(sys.argv[1:] + ['--mode', 'train', '--db', 'path/to/db']))
