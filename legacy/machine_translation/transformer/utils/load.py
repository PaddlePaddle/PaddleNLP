import pickle
import six
import warnings
from functools import partial

import paddle.fluid as fluid


def load(program, model_path, executor=None, var_list=None):
    """
    To load python2 saved models in python3.
    """
    try:
        fluid.load(program, model_path, executor, var_list)
    except UnicodeDecodeError:
        warnings.warn(
            "An UnicodeDecodeError is catched, which might be caused by loading "
            "a python2 saved model. Encoding of pickle.load would be set and "
            "load again automatically.")
        if six.PY3:
            load_bak = pickle.load
            pickle.load = partial(load_bak, encoding="latin1")
            fluid.load(program, model_path, executor, var_list)
            pickle.load = load_bak
