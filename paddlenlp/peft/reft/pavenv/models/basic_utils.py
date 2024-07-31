# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import logging
import os
import random

import numpy as np
import paddle


# Introducing corresponding classes based on strings
def get_type_from_string(type_str):
    """Help function to convert string to type"""
    # Remove <class ' and '> from the string
    type_str = type_str.replace("<class '", "").replace("'>", "")

    # Split the string into module and class name
    module_name, class_name = type_str.rsplit(".", 1)

    # Import the module
    if not module_name.startswith("paddlenlp"):
        module_name = f"paddlenlp.peft.reft.{module_name}"
    module = importlib.import_module(module_name)

    # Get the class
    cls = getattr(module, class_name)

    return cls


def create_directory(path):
    """Create directory if not exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Directory '{path}' created successfully.")
    else:
        logging.info(f"Directory '{path}' already exists.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def count_parameters(model):
    """Count parameters of a model that require gradients"""
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)
