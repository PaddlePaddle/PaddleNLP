import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import operator
import numpy as np
from paddlenlp.peft.lora.lora_quantization_layers import QuantizationLoRALinear, QuantizationLoRABaseLinear
from functools import reduce  # Required in Python 3
from scipy.stats import norm
from paddleslim.lc.quantizers.quant_func import create_dynamic_map
from paddleslim.lc.layers.linear import Linear4bit
from paddlenlp.quantization.qlora import qlora_weight_dequantize, qlora_weight_quantize

cache_folder_path = ''
module_num = 0
sigma = 1 / norm.ppf(paddle.linspace(0.9677083, 0.5, 9)[:-1]).tolist()[0]

def get_my_model(model, model_fp, blocksize2=256, tau_range=0.1, tau_n=25):
    model.model = _replace_with_ours_lora_4bit_linear(model.model, model_fp=model_fp, blocksize2=blocksize2, tau_range=tau_range, tau_n=tau_n)
    return model

def prod(iterable):
    return reduce(operator.mul, iterable, 1)
    
normal_map_fp8 = create_dynamic_map()
normal_map_fp8 = paddle.to_tensor(normal_map_fp8)
def quantize_tensor(X, L, idx=False):
    X_shape = X.shape
    X_expanded = X.reshape([-1, 1])
    L_reshaped = L.reshape([1, -1])
    abs_diff = paddle.abs(X_expanded - L_reshaped)
    min_index = paddle.argmin(abs_diff, axis=-1)
    min_index = min_index.cast("uint8").reshape(X_shape)
    return min_index

def dequantize_tensor(X, L):
    return paddle.index_select(L, axis=0, index=paddle.to_tensor(X, dtype=paddle.int32).reshape([-1])).reshape(X.shape)

@paddle.no_grad()
def nf4_quant(weight, weight_shape, tau, state, quant_algo):
    weight = weight.reshape([-1, 256, 64])
    tau = tau.reshape([-1, 256, 1])
    _weight = (weight - tau).reshape(weight_shape)
    nf4_weight = qlora_weight_quantize(_weight.cuda(), quant_algo, double_quant=True)
    tau2 = tau.abs().max(axis=1, keepdim=True)[0]
    tau1 = quantize_tensor(tau / tau2, normal_map_fp8)
    return nf4_weight, tau1.reshape([-1, 256]), tau2.reshape([-1, 1])

@paddle.no_grad()
def evaluate_entropy(weight_int8, blocksize):
    _weight_int8 = weight_int8.reshape([-1, 1])
    weight_nf4 = paddle.concat((_weight_int8//16, _weight_int8 & paddle.to_tensor(15).cast('uint8')), 1).reshape([1, -1, blocksize])
    weight_nf4_repeat = weight_nf4.cast('int32').tile([16, 1, 1])
    values = paddle.to_tensor(list(range(16))).reshape([16, 1, 1])
    freqs = (weight_nf4_repeat==values.cast('int32')).sum(axis=-1, keepdim=True) / blocksize
    entropy = -freqs * paddle.log2(freqs)
    entropy = paddle.where(paddle.isnan(entropy), 0., entropy)
    entropy = entropy.sum(axis=0)
    return entropy

@paddle.no_grad()
def search(fp_weight: paddle.Tensor, fp_weight_shape, state, quant_algo, tau_range=0.1, tau_n=25, blocksize=64, blocksize2=256):
    fp_weight = fp_weight.reshape([-1, blocksize2, blocksize])
    tau0 = fp_weight.median(2, keepdim=True)[0] # [-1, 256, 1]
    absmax = (fp_weight - tau0).abs().max(2, keepdim=True)[0]
    
    entropy_max, factor_best = None, None
    for factor in np.linspace(-tau_range*sigma, tau_range*sigma, tau_n*2+1):
        tau = factor * absmax + tau0
        nf4_weight, _, _ = nf4_quant(fp_weight, fp_weight_shape, tau, state, quant_algo)
        entropy = evaluate_entropy(nf4_weight['quant_weight'], blocksize)
        
        if entropy_max is None:
            entropy_max = entropy
            factor_best = paddle.full_like(entropy, factor)
        else:
            factor_best = paddle.where(entropy > entropy_max, factor, factor_best)
            entropy_max = paddle.maximum(entropy_max, entropy)
    
    tau = factor_best.reshape([-1, 256, 1]).cast('float32') * absmax + tau0
    nf4_weight, tau1, tau2 = nf4_quant(fp_weight, fp_weight_shape, tau, state, quant_algo)
    return nf4_weight, tau1, tau2

class IRQuantizationLoRALinear(QuantizationLoRALinear):
    def __init__(
        self, old_model, model_fp=None, blocksize2=256, tau_range=0.1, tau_n=51
    ):
        for key, value in old_model.__dict__.items():
            setattr(self, key, value)

        fp_weight = model_fp.weight.data.contiguous().cpu()
        fp_weight_shape = fp_weight.shape
        
        quant_weight, quant_dtype, quantization_config, weight_quantize_algo, dtype, quant_scale, quant_state = self.quant_weight, self.quant_dtype, self.quantization_config, self.weight_quantize_algo, self._dtype, self.quant_scale, None

        del model_fp

        nf4_weight, tau1, tau2 = search(
            fp_weight=fp_weight,
            fp_weight_shape=fp_weight_shape,
            state=quant_state,
            quant_algo=weight_quantize_algo,
            tau_range=tau_range, tau_n=tau_n,
            blocksize2=blocksize2
        )
        
        self.quant_weight.data = nf4_weight['quant_weight']
        self.qquant_scale = nf4_weight['qquant_scale']
        self.double_quant_scale = nf4_weight['double_quant_scale']
        self.quant_sacle_offset = nf4_weight['quant_sacle_offset']
        self.tau_quant, self.tau_absmax = tau1, tau2
        self.fp_weight_shape = fp_weight_shape
        
        del fp_weight, nf4_weight
        
        a = paddle.zeros(1)
        self.lora_default_A_scale = paddle.create_parameter(shape=a.shape, dtype=a.dtype, default_initializer=nn.initializer.Assign(a))
        self.lora_default_B_scale = paddle.create_parameter(shape=a.shape, dtype=a.dtype, default_initializer=nn.initializer.Assign(a))
        
    def forward(self, x: paddle.Tensor):
                
        with paddle.no_grad():
            fp_B = qlora_weight_dequantize(self.quant_weight, self.weight_quantize_algo, (self.qquant_scale, self.double_quant_scale.cast('float32'), self.quant_sacle_offset.cast('float32')), double_quant=True)
            tau = (dequantize_tensor(self.tau_quant, normal_map_fp8).reshape([-1, 256, 1]) * self.tau_absmax.reshape([-1, 1, 1]))
            blocksize = paddle.prod(paddle.to_tensor(fp_B.shape)) // paddle.prod(paddle.to_tensor(tau.shape))
            fp_B = (fp_B.reshape([-1, blocksize.item()]) + tau.reshape([-1, 1])).reshape(self.fp_weight_shape)
        
        result = paddle.nn.functional.linear(x, fp_B, self.bias)

        if not self.disable_lora:
            x1 = self.lora_dropout(x)
            x2 = x1 @ self.lora_A + self.lora_default_A_scale * x.reshape([_ for _ in x.shape[:-1]] + [self.lora_A.shape[-1]] + [-1]).mean(axis=-1)
            x3 = ((x2 @ self.lora_B).reshape([_ for _ in x2.shape] + [-1]) + (self.lora_default_B_scale * x2.unsqueeze(-1))).reshape([_ for _ in x2.shape[:-1]] + [-1])
            result += x3 * self.scaling
            
        return result

def _replace_with_ours_lora_4bit_linear(
    model: nn.Layer, current_key_name=None, model_fp=None, blocksize2=256, tau_range=0.5, tau_n=51
):
    assert isinstance(model_fp, nn.Layer)
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, QuantizationLoRALinear):
            _modules = dict(model_fp.named_sublayers())

            print(name)
            new_layer = IRQuantizationLoRALinear(dict(model.named_sublayers())[name], model_fp=dict(model_fp.named_sublayers())[name], blocksize2=blocksize2, tau_range=tau_range, tau_n=tau_n)
            setattr(model, name, new_layer)
        
        if len(list(module.children())) > 0:
            _modules = dict(model_fp.named_sublayers())
            if name in _modules.keys():
                _ = _replace_with_ours_lora_4bit_linear(
                    module,
                    current_key_name, _modules[name], blocksize2, tau_range, tau_n
                )
            else:
                _ = _replace_with_ours_lora_4bit_linear(
                    module,
                    current_key_name, None, blocksize2, tau_range, tau_n
                )
        current_key_name.pop(-1)
    return model
