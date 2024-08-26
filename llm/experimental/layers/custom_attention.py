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
"""
Custome Attention Layer for quantization.
"""
import paddle
from paddle.nn import Layer
import paddle.tensor as tensor
from paddle.nn.quant.format import ConvertibleQuantedLayer

class QuantizedCustomAttentionLayer(ConvertibleQuantedLayer):
    """
    Quantized Custom Attention Layer.
    """
    def __init__(self, layer: Layer, q_config=None):
        """
            Initialize the QuantizeWrapper class.
        
            Args:
                layer (Layer): The layer to be quantized.
                q_config (QuantConfig, optional): The quantization configuration. Defaults to None.
        """
        super().__init__()
        # hard code: get activation quanter from weight 
        self.activation_quanter_k = q_config.weight._instance(layer)
        self.activation_quanter_v = q_config.activation._instance(layer)
        self.layer = layer
        self.enable_fake_quant = False
        self.quant_info = None
        layer_name = self.layer.full_name()
        self.layer_id = int(layer_name.split("_")[-1])
        self.kv_losses = {}
        
    def forward(self,
        q,
        config,
        k,
        v,
        attention_mask,
        output_attentions,
        # alibi,
        # attn_mask_startend_row_indices,
        # sequence_parallel, 
        **kwargs):
        """forward"""
        # import pdb;pdb.set_trace()
        if self.enable_fake_quant:
            self.collect_kv_quant_policy(q, k, v, **kwargs)
        perm = [0, 2, 1, 3]  # [1, 2, 0, 3] if self.sequence_parallel else [0, 2, 1, 3]
        tmp_k = tensor.transpose(x=k, perm=perm)
        tmp_v = tensor.transpose(x=v, perm=perm)
        if self.activation_quanter_k is not None:
            tmp_k = self.activation_quanter_k(tmp_k)
        if self.activation_quanter_v is not None:
            tmp_v = self.activation_quanter_v(tmp_v)  
        k = tensor.transpose(x=tmp_k, perm=perm)
        v = tensor.transpose(x=tmp_v, perm=perm)
        return self.layer(
            q,
            config,
            k,
            v,
            attention_mask,
            output_attentions,
            # alibi,
            # attn_mask_startend_row_indices,
            # sequence_parallel, 
            **kwargs)

    def weights_to_quanters(self):
        """weights to quanters"""
        return []

    def activation_quanters(self):
        """activation to quanters"""
        return ['activation_quanter_k', 'activation_quanter_v']
    
    # def cal_scales_zps(self, min, max, quant_bits):
    #     """calculate scale and zero point"""
    #     bnt = (1 << (quant_bits - 1)) - 1
    #     qmin = -bnt - 1
    #     qmax = bnt
    #     scales = (max - min) / float(qmax - qmin)
    #     zps = qmin - paddle.round(min / scales)
    #     zps = paddle.clip(zps, qmin, qmax)
    #     return scales, zps
    
    # def fake_quant_dequant(self, input, scales, zps, quant_bits):
    #     """fake quant dequant"""
    #     scales = scales.cast("float32")
    #     zps = zps.cast("float32")
    #     bnt = (1 << (quant_bits - 1)) - 1
    #     quant_weight = paddle.clip(
    #                 paddle.round(input.cast("float32") / scales) + zps,
    #                 -bnt - 1,
    #                 bnt,
    #             )

    #     dequant_weight = ((quant_weight - zps) * scales).cast(input.dtype)
    #     return dequant_weight
    
    # def collect_kv_quant_policy(self, q, k, v, **kwargs):
    #     """collect kv quant policy for different options"""
    #     quant_info_layer = self.quant_info.get(self.layer_id)
    #     k_min = quant_info_layer.get('k_min')
    #     k_max = quant_info_layer.get('k_max')
    #     k_min_global = quant_info_layer.get('k_min_global')
    #     k_max_global = quant_info_layer.get('k_max_global')
    #     v_min = quant_info_layer.get('v_min')
    #     v_max = quant_info_layer.get('v_max')
    #     v_min_global = quant_info_layer.get('v_min_global')
    #     v_max_global = quant_info_layer.get('v_max_global')
    #     search_iter = quant_info_layer.get('search_iter')

    #     int4_clib = []
    #     ratio_list = [-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1]
    #     out_qkq, _ = self.layer(q, k, v, **kwargs)

    #     for option_calib in ratio_list:
    #         k_max_cur = k_max * option_calib + k_max_global * (1 - option_calib)
    #         k_min_cur = k_min * option_calib + k_min_global * (1 - option_calib)

    #         v_max_cur = v_max * option_calib + v_max_global * (1 - option_calib)
    #         v_min_cur = v_min * option_calib + v_min_global * (1 - option_calib)

    #         k_scales_4, k_zps_4 = self.cal_scales_zps(k_min_cur, k_max_cur, quant_bits=4)
    #         v_scales_4, v_zps_4 = self.cal_scales_zps(v_min_cur, v_max_cur, quant_bits=4)

    #         out_int4, _ = self.fake_attn_helper(q, k, v, k_scales_4, k_zps_4, 
    #                                             v_scales_4, v_zps_4, quant_bits=4, **kwargs)
    #         loss_kv_cur = ((out_qkq - out_int4) ** 2).mean()
    #         kv_name = "kv_layer_" + str(self.layer_id)

    #         if kv_name not in self.kv_losses.keys(): self.kv_losses[kv_name] = {}
    #         if option_calib not in self.kv_losses[kv_name]:
    #             self.kv_losses[kv_name][option_calib] = loss_kv_cur
    #         else:
    #             self.kv_losses[kv_name][option_calib] = (
    #                 self.kv_losses[kv_name][option_calib] * (search_iter - 1) + loss_kv_cur) / search_iter


    # def fake_attn_helper(self, q, k, v, k_scales, k_zps, v_scales, v_zps, quant_bits, **kwargs):
    #     """fake attention calculated by fake quant dequant key and value """
    #     perm = [0, 2, 1, 3]  # [1, 2, 0, 3] if self.sequence_parallel else [0, 2, 1, 3]
    #     tmp_k = tensor.transpose(x=k, perm=perm)
    #     tmp_v = tensor.transpose(x=v, perm=perm)
    #     k_qdq = self.fake_quant_dequant(tmp_k, k_scales, k_zps, quant_bits)
    #     v_qdq = self.fake_quant_dequant(tmp_v, v_scales, v_zps, quant_bits)
    #     k_qdq = tensor.transpose(x=k_qdq, perm=perm)
    #     v_qdq = tensor.transpose(x=v_qdq, perm=perm)
    #     out_qkq = self.layer(q, k_qdq, v_qdq, **kwargs)
    #     return out_qkq
