# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY

from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle.jit.marker import unified

@unified
def fused_rms_norm(x,scale,epsilon):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("fused_rms_norm", x,scale,epsilon)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res)==1 else res
    else:
        ins = {}
        ins_map = {'x' : x,'scale' : scale}
        outs = {}
        outs_list = ['y','invvar']
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fused_rms_norm", **locals())

        outs['y'] = helper.create_variable(dtype='float32')
        outs['invvar'] = helper.create_variable(dtype='float32')
        helper.append_op(type="fused_rms_norm", inputs=ins, outputs=outs, attrs={'epsilon' : epsilon})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res)==1 else res


from paddle import _C_ops
from paddle.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle.jit.marker import unified

@unified
def fused_ln(x,scale,bias,epsilon):
    # The output variable's dtype use default value 'float32',
    # and the actual dtype of output variable will be inferred in runtime.
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op("fused_ln", x,scale,bias,epsilon)
        res = []
        start_idx = 0
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        res.append(outs[start_idx])
        start_idx += 1
        return res[0] if len(res)==1 else res
    else:
        ins = {}
        ins_map = {'x' : x,'scale' : scale,'bias' : bias}
        outs = {}
        outs_list = ['y','mean','invvar']
        for key, value in ins_map.items():
            # handle optional inputs
            if value is not None:
                ins[key] = value
        helper = LayerHelper("fused_ln", **locals())

        outs['y'] = helper.create_variable(dtype='float32')
        outs['mean'] = helper.create_variable(dtype='float32')
        outs['invvar'] = helper.create_variable(dtype='float32')
        helper.append_op(type="fused_ln", inputs=ins, outputs=outs, attrs={'epsilon' : epsilon})
        res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
        return res[0] if len(res)==1 else res


import os
import sys
import types
import paddle
import importlib.abc
import importlib.util

cur_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(cur_dir, "lib/fused_ln_pd.so")

def __bootstrap__():
    assert os.path.exists(so_path)
    # load custom op shared library with abs path
    custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # Cpp Extension only support Linux now
        mod = types.ModuleType(__name__)
    else:
        try:
            spec = importlib.util.spec_from_file_location(__name__, so_path)
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(mod)
        except ImportError:
            mod = types.ModuleType(__name__)

    for custom_op in custom_ops:
        setattr(mod, custom_op, eval(custom_op))

__bootstrap__()

