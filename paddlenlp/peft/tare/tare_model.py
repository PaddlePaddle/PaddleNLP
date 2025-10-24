# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import re
import time

import paddle
import paddle.nn.functional as F


def cast_all_params_(layer: paddle.nn.Layer, dtype):
    """把整模里所有 Parameter 的 dtype 原地转成 dtype。"""
    for _, p in layer.named_parameters():
        if p.dtype != dtype:
            p.set_value(p.astype(dtype))


def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f"gpu:{type}"
    elif isinstance(type, str):
        if "cuda" in type:
            type = type.replace("cuda", "gpu")
        if "cpu" in type:
            type = "cpu"
        elif index is not None:
            type = f"{type}:{index}"
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = "cpu"
    elif isinstance(type, paddle.CUDAPlace):
        type = f"gpu:{type.get_device_id()}"

    return type


def _get_token(obj, tok: str):
    """
    根据 tok 从 obj 里取下一层：
    - name[idx] 形式：先 getattr(obj, name) 再 [idx]
    - 纯数字：认为 obj 是 list-like，直接 [int(tok)]
    - 普通标识符：getattr(obj, tok)
    """
    m = re.fullmatch(r"([A-Za-z_]\w*)\[(\d+)\]", tok)
    if m:
        name, idx = m.group(1), int(m.group(2))
        container = getattr(obj, name)
        return container[idx]
    if tok.isdigit():
        return obj[int(tok)]
    return getattr(obj, tok)


def _resolve_parent_and_name(root, path: str):
    """
    沿着点路径解析，返回 (parent, last_name_or_index_str, obj)
    例如 "llama.layers.0.mlp.down_proj" ->
      parent=<mlp Layer>, last="down_proj", obj=<Linear>
    """
    parts = path.split(".") if path else []
    if not parts:
        raise AttributeError(f"Empty path for resolve: {path}")
    cur = root
    parent = None
    last = None
    for tok in parts:
        parent = cur
        last = tok
        cur = _get_token(cur, tok)
    return parent, last, cur


def _set_child(parent, name_or_idx: str, new_obj):
    """
    把 parent 的子模块替换为 new_obj。
    - 如果 name_or_idx 是数字，认为 parent 是 list-like，做 parent[int] = new_obj
    - 否则 setattr(parent, name_or_idx, new_obj)
    """
    print("parent:", parent)
    print("name_or_idx:", name_or_idx)
    print("new_obj", new_obj)
    if name_or_idx.isdigit():
        print("Yes!!!!!!")
        parent[int(name_or_idx)] = new_obj
    else:
        setattr(parent, name_or_idx, new_obj)


device = device2str("cuda" if paddle.device.cuda.device_count() >= 1 else "cpu")
target_dict = {}
total_parameter = 0
paddle.set_device(device)


class _DeltaUnit(paddle.nn.Layer):
    def __init__(self, hidden_size, dtype):
        super().__init__()
        self.activation_scaling = self.create_parameter(
            shape=[1, hidden_size],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.activation_bias = self.create_parameter(
            shape=[1, hidden_size],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )


class ActivationLayer(paddle.nn.Layer):
    def __init__(self, hidden_size, update_layer, layer_type="all", op_position="ffn", is_llama=False, n=8, k=7):
        super().__init__()
        self.update_layer = update_layer
        self.layer_type = layer_type
        self.op_position = op_position
        self.n = n
        self.k = k
        self.fc = paddle.nn.Linear(in_features=5120, out_features=self.n)
        if is_llama:
            self.weight_type = paddle.bfloat16
        else:
            self.weight_type = paddle.float16
        if self.layer_type == "all":
            self.delta_vector = paddle.nn.LayerList(
                _DeltaUnit(hidden_size, dtype=self.weight_type) for _ in range(self.n)
            )
        elif self.layer_type == "scaling":
            self.delta_vector = paddle.nn.ParameterDict(
                parameters={"activation_scaling": paddle.nn.parameter.Parameter(paddle.ones(1, hidden_size))}
            )
        elif self.layer_type == "bias":
            self.delta_vector = paddle.nn.ParameterDict(
                parameters={"activation_bias": paddle.nn.parameter.Parameter(paddle.zeros(1, hidden_size))}
            )
        elif self.layer_type == "ln":
            self.delta_vector = paddle.nn.ParameterDict(
                parameters={
                    "activation_ln": paddle.nn.LayerNorm(normalized_shape=hidden_size),
                    "activation_scaling": paddle.nn.parameter.Parameter(paddle.ones(1, hidden_size)),
                    "activation_bias": paddle.nn.parameter.Parameter(paddle.zeros(1, hidden_size)),
                }
            )
        self.weight = paddle.rand(shape=[1], dtype=self.weight_type)

    def forward(self, x, input_tensor=None):
        # self.delta_vector.to(self.weight_type).to(device)
        if self.op_position == "res" or self.op_position == "res_with_attn" or self.op_position == "res_with_res":
            hidden_states = self.update_layer(x, input_tensor)
        else:
            hidden_states = self.update_layer(x)
        if self.layer_type == "all":
            hidden_states_new = self.fc(hidden_states)
            # hidden_states_updated = hidden_states.clone()
            # print('hidden_states_new:',hidden_states_new.shape)
            topk_values, topk_indices = paddle.topk(hidden_states_new, self.k, axis=-1)
            # topk_weights = paddle.softmax(topk_values, axis=-1)
            topk_weights = F.softmax(topk_values, axis=-1)

            scaling_tensor = paddle.stack([v.activation_scaling for v in self.delta_vector], axis=0).squeeze(
                1
            )  # [8, hidden]
            bias_tensor = paddle.stack([v.activation_bias for v in self.delta_vector], axis=0).squeeze(
                1
            )  # [8, hidden]

            hidden_states_updated = paddle.zeros_like(hidden_states)  # [..., hidden]
            for i in range(self.k):
                idx = topk_indices[..., i]  # [...,]
                s_i = scaling_tensor[idx]  # [..., hidden]
                b_i = bias_tensor[idx]  # [..., hidden]
                hd_i = hidden_states * s_i + b_i  # [..., hidden]
                w_i = topk_weights[..., i : i + 1]  # [..., 1]
                hidden_states_updated = hidden_states_updated + hd_i * w_i  # 广播到 [..., hidden]

            hidden_states = hidden_states_updated.squeeze(-1)
        elif self.layer_type == "scaling":
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
        elif self.layer_type == "bias":
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
        elif self.layer_type == "ln":
            hidden_states = hidden_states * self.delta_vector["activation_scaling"]
            hidden_states = hidden_states + self.delta_vector["activation_bias"]
            hidden_states = self.delta_vector["activation_ln"](hidden_states)
        if self.op_position == "res_with_res":
            hidden_states = hidden_states + x
        return hidden_states


class TAREModel(paddle.nn.Layer):
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, base_model, op_position="ffn", layer_type="all", exclude_layers=[], only_lora=False, n=8, k=7):
        print("初始化模型！！！！！！")
        time.sleep(5)
        super().__init__()
        self.base_model = base_model
        self.model_type = "llama-7b"
        self.layer_type = layer_type
        self.op_position = op_position
        self.exclude_layers = exclude_layers
        self.n = n
        self.k = k
        if exclude_layers:
            pattern_str = "|".join(map(str, exclude_layers))
            pattern = re.compile("\\b(?:" + pattern_str + ")\\b")
        self.frozen_model()
        if only_lora is False:
            key_list = [key for key, _ in base_model.named_sublayers(include_self=True)]
            for key in key_list:
                if exclude_layers:
                    match = pattern.search(key)
                    if match:
                        continue
                if self.check_update(key):
                    self.replace_layer(key)
        print(self.print_trainable_parameters())

    def check_update(self, key):
        if self.op_position == "ffn":
            return self.match_substring(key)

    def generate(self, **args):
        return self.base_model.generate(**args)

    def replace_layer(self, key):
        # 打印一下，便于定位真实前缀（有的模型是 self.llama，有的是 self.model）
        print("base_model:", self.base_model)
        print("key:", key)

        # 逐段解析，拿到 旧模块 与 其父模块
        parent, last_name, old_mod = _resolve_parent_and_name(self.base_model, key)
        print("replaced_module(old):", old_mod)

        # 构造你的替换模块；确保 forward 签名与 old_mod 兼容
        new_module = ActivationLayer(
            hidden_size=self.base_model.config.hidden_size,
            update_layer=old_mod,
            layer_type=self.layer_type,
            op_position=self.op_position,
            is_llama=True,
            n=self.n,
            k=self.k,
        )

        # 覆盖到父模块上
        _set_child(parent, last_name, new_module)

    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0

        for name, param in self.base_model.named_parameters():
            # 统计参数量
            try:
                n = int(paddle.numel(param))
            except Exception:
                # 兜底
                n = 1
                for d in param.shape:
                    n *= int(d)
            total_parameters += n

            # 在 Paddle 中，用 stop_gradient 判断是否可训练
            is_trainable = not getattr(param, "stop_gradient", True)
            if is_trainable:
                trainable_parameters += n

        pct = (100.0 * trainable_parameters / total_parameters) if total_parameters else 0.0
        return {
            "total_para:": total_parameters,
            "trainable_para: ": trainable_parameters,
            "trainable%:": f"{pct:.4f}",
        }

    def frozen_model(self):
        for name, param in self.base_model.named_parameters():
            if "lora" in name:
                param.stop_gradient = not True
            else:
                param.stop_gradient = not False

    def match_substring(self, input_string):
        pattern = "down_proj"
        pattern2 = "c_proj"
        match = re.search(pattern, input_string) or re.search(pattern2, input_string)
        if match:
            return True
        else:
            return False

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        output = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, position_ids=position_ids
        )
        return output

    def load_model(self, save_path):
        save_sd = paddle.load(str(save_path))
        model_sd = self.base_model.state_dict()
        loaded = 0
        for k, v in save_sd.items():
            # print(k)
            if k in model_sd:
                tgt = model_sd[k]
                if v.dtype != tgt.dtype:
                    v = v.astype(tgt.dtype)
                assert list(v.shape) == list(tgt.shape), f"{k} shape {v.shape} vs {tgt.shape}"
                tgt.set_value(v)
                tgt.stop_gradient = False  # 若这些参数要训练
                loaded += 1
        print(f"loaded {loaded} tensors")
        time.sleep(5)

    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        for k in state_dict:
            if "activation_" in k or "lora" in k or "fc" in k:
                print(k)

        save_dict = {k: state_dict[k] for k in state_dict if "activation_" in k or "lora" in k or "fc" in k}
        return save_dict

    def save_model(self, save_path):
        save_dict = self.get_save_dict()
        paddle.save(obj=save_dict, path=save_path)
