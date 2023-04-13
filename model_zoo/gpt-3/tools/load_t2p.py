# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

def trans_model_for_non_pp(torch_state, num_layer=24):

    # model_state
    torch_model_state = torch_state.pop('model') \
        if 'model' in torch_state.keys() else torch_state
    paddle_model_state = {}
    paddle_model_state["gpt.embeddings.position_embeddings.weight"] = torch_model_state[
        "language_model"]["embedding"]["position_embeddings"]["weight"]
    paddle_model_state["gpt.embeddings.word_embeddings.weight"] = torch_model_state[
        "language_model"]["embedding"]["word_embeddings"]["weight"]
    maps = []
    for i in range(num_layer):
        maps.append([
            f"gpt.decoder.layers.{i}.linear1.bias",
            f"layers.{i}.mlp.dense_h_to_4h.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.linear1.weight",
            f"layers.{i}.mlp.dense_h_to_4h.weight", "T"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.linear2.bias",
            f"layers.{i}.mlp.dense_4h_to_h.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.linear2.weight",
            f"layers.{i}.mlp.dense_4h_to_h.weight", "T"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.norm1.bias",
            f"layers.{i}.input_layernorm.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.norm1.weight",
            f"layers.{i}.input_layernorm.weight"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.norm2.bias",
            f"layers.{i}.post_attention_layernorm.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.norm2.weight",
            f"layers.{i}.post_attention_layernorm.weight"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.self_attn.out_proj.bias",
            f"layers.{i}.self_attention.dense.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.self_attn.out_proj.weight",
            f"layers.{i}.self_attention.dense.weight", "T"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.self_attn.qkv_proj.bias",
            f"layers.{i}.self_attention.query_key_value.bias"
        ])
        maps.append([
            f"gpt.decoder.layers.{i}.self_attn.qkv_proj.weight",
            f"layers.{i}.self_attention.query_key_value.weight", "T"
        ])

    maps.append([f"gpt.decoder.norm.weight", f"final_layernorm.weight"])
    maps.append([f"gpt.decoder.norm.bias", f"final_layernorm.bias"])

    for m in maps:
        if len(m) == 2:
            paddle_model_state[m[0]] = torch_model_state["language_model"]["encoder"][m[
                1]]
        else:
            paddle_model_state[m[0]] = torch_model_state["language_model"]["encoder"][m[
                1]].T

    for k in paddle_model_state.keys():
        paddle_model_state[k] = paddle_model_state[k].float().numpy()

    return paddle_model_state

def trans_model_for_pp(torch_state, pp_rank=0, pp_size=1, num_layer=24, max_position_embed=1024, hidden_size=1024):

    # model_state
    torch_model_state = torch_state.pop('model') \
        if 'model' in torch_state.keys() else torch_state
    paddle_model_state = {}

    assert pp_size > 1 and pp_rank < pp_size
    num_layer = int(num_layer / pp_size)

    from paddle.distributed import fleet
    hcg = fleet.get_hybrid_communicate_group()

    if pp_rank == 0:
        paddle_model_state["model.shared_layers.embed.word_embeddings.weight"] = torch_model_state[
            "language_model"]["embedding"]["word_embeddings"]["weight"]
        paddle_model_state["model.shared_layers.embed.position_embeddings.weight"] = torch_model_state[
            "language_model"]["embedding"]["position_embeddings"]["weight"]
        print("torch:")
        print(torch_model_state["language_model"]["embedding"]["word_embeddings"]["weight"].shape)
        print("paddle:")
        print(paddle_model_state["model.shared_layers.embed.word_embeddings.weight"].shape)
    elif pp_rank == pp_size - 1:
        paddle_model_state["model.shared_layers.embed.word_embeddings.weight"] = torch_model_state["word_embeddings_for_head"]["weight"]
        print("torch:")
        print(torch_model_state["word_embeddings_for_head"]["weight"].shape)
        print("paddle:")
        print(paddle_model_state["model.shared_layers.embed.word_embeddings.weight"].shape)


    if pp_rank == 0:
        paddle.distributed.broadcast(
            paddle.to_tensor(paddle_model_state["model.shared_layers.embed.position_embeddings.weight"].float().numpy()),
            group=hcg.get_pipe_parallel_group(),
            src=hcg.get_pipe_parallel_group().ranks[0],
        )
    else:
        tmp_tensor = paddle.zeros(shape=[max_position_embed, hidden_size])
        paddle.distributed.broadcast(
            tmp_tensor,
            group=hcg.get_pipe_parallel_group(),
            src=hcg.get_pipe_parallel_group().ranks[0],
        )

    if pp_rank > 0 and pp_rank < pp_size - 1:
        tmp_tensor._clear_data()
    elif pp_rank == pp_size - 1:
        paddle_model_state["model.shared_layers.embed.position_embeddings.weight"] = tmp_tensor.numpy()

    maps = []
    for i in range(num_layer):
        j = i + pp_rank * num_layer + 1
        maps.append([
            f"model.{j}.linear1.bias",
            f"layers.{i}.mlp.dense_h_to_4h.bias"
        ])
        maps.append([
            f"model.{j}.linear1.weight",
            f"layers.{i}.mlp.dense_h_to_4h.weight", "T"
        ])
        maps.append([
            f"model.{j}.linear2.bias",
            f"layers.{i}.mlp.dense_4h_to_h.bias"
        ])
        maps.append([
            f"model.{j}.linear2.weight",
            f"layers.{i}.mlp.dense_4h_to_h.weight", "T"
        ])
        maps.append([
            f"model.{j}.norm1.bias",
            f"layers.{i}.input_layernorm.bias"
        ])
        maps.append([
            f"model.{j}.norm1.weight",
            f"layers.{i}.input_layernorm.weight"
        ])
        maps.append([
            f"model.{j}.norm2.bias",
            f"layers.{i}.post_attention_layernorm.bias"
        ])
        maps.append([
            f"model.{j}.norm2.weight",
            f"layers.{i}.post_attention_layernorm.weight"
        ])
        maps.append([
            f"model.{j}.self_attn.out_proj.bias",
            f"layers.{i}.self_attention.dense.bias"
        ])
        maps.append([
            f"model.{j}.self_attn.out_proj.weight",
            f"layers.{i}.self_attention.dense.weight", "T"
        ])
        maps.append([
            f"model.{j}.self_attn.qkv_proj.bias",
            f"layers.{i}.self_attention.query_key_value.bias"
        ])
        maps.append([
            f"model.{j}.self_attn.qkv_proj.weight",
            f"layers.{i}.self_attention.query_key_value.weight", "T"
        ])

    if pp_rank == pp_size - 1:
        maps.append([f"model.{pp_size * num_layer + 1}.norm.weight", f"final_layernorm.weight"])
        maps.append([f"model.{pp_size * num_layer + 1}.norm.bias", f"final_layernorm.bias"])

    for m in maps:
        if len(m) == 2:
            paddle_model_state[m[0]] = torch_model_state["language_model"]["encoder"][m[
                1]]
        else:
            paddle_model_state[m[0]] = torch_model_state["language_model"]["encoder"][m[
                1]].T

    for k in paddle_model_state.keys():
        paddle_model_state[k] = paddle_model_state[k].float().numpy() if not isinstance(
            paddle_model_state[k], np.ndarray) else paddle_model_state[k]

    return paddle_model_state

def trans_optim(torch_state, paddle_model, pp_rank=0, pp_size=1, num_layer=24):
    torch_optim_state = torch_state['optimizer']
    list_params_decay = []
    list_params_no_decay = []
    paddle_optim_state = {}

    param_list = paddle_model.parameters()

    # stage[0] embedding
    if pp_rank == 0:
        list_params_decay.append(param_list.pop(0)) # word_embedding
        list_params_decay.append(param_list.pop(0)) # position_embedding
    elif pp_rank == pp_size - 1:
        shared_word_embedding = param_list.pop(0)
        param_list.pop(0)
    
    #  00. linear_0.w_0: Tensor                            layers.0.input_layernorm.weight
    #  01. linear_0.b_0: Tensor                            layers.0.input_layernorm.bias
    #  02. linear_1.w_0: Tensor                            layers.0.self_attention.query_key_value.weight
    #  03. linear_1.b_0: Tensor                            layers.0.self_attention.query_key_value.bias
    #  04. linear_2.w_0: Tensor                            layers.0.self_attention.dense.weight
    #  05. linear_2.b_0: Tensor             =>             layers.0.self_attention.dense.bias
    #  06. linear_3.w_0: Tensor                            layers.0.post_attention_layernorm.weight
    #  07. linear_3.b_0: Tensor                            layers.0.post_attention_layernorm.bias
    #  08. layer_norm_0.w_0: Tensor                        layers.0.mlp.dense_h_to_4h.weight
    #  09. layer_norm_0.b_0: Tensor                        layers.0.mlp.dense_h_to_4h.bias
    #  10. layer_norm_1.w_0: Tensor                        layers.0.mlp.dense_4h_to_h.weight
    #  11. layer_norm_1.b_0: Tensor                        layers.0.mlp.dense_4h_to_h.bias

    for index in range(int(num_layer / pp_size)):
        layer_params = param_list[12 * index : 12 * index + 8]
        layer_params.insert(0, param_list[12 * index + 8]) # input_layernorm.weight
        layer_params.insert(1, param_list[12 * index + 9]) # input_layernorm.bias
        layer_params.insert(6, param_list[12 * index + 10]) # post_attention_layernorm.weight
        layer_params.insert(7, param_list[12 * index + 11]) # post_attention_layernorm.bias

        for param in layer_params:
            if any(nd in param.name for nd in ["bias", "norm", "b_0"]):
                list_params_no_decay.append(param)
            else:
                list_params_decay.append(param)
    
    if pp_rank == pp_size - 1:
        list_params_decay.append(shared_word_embedding)
        list_params_no_decay.append(param_list.pop(-2)) # final_layernorm.weight
        list_params_no_decay.append(param_list.pop(-1)) # final_layernorm.bias

    paddle_params_list = list_params_decay + list_params_no_decay

    for index, param in enumerate(paddle_params_list):

        moment1 = paddle.to_tensor(torch_optim_state['optimizer']['state'][index]['exp_avg'].numpy(), dtype=paddle.float32) if "embedding" in param.name or len(param.shape) == 1 else paddle.to_tensor(torch_optim_state['optimizer']['state'][index]['exp_avg'].numpy().T, dtype=paddle.float32)
        assert moment1.shape == param.shape, "m1.shape : {}     param.shape : {}      param.name : {}".format(moment1.shape, param.shape, param.name)
        paddle_optim_state["{}_fp32_master_0_moment1_0".format(param.name)] = moment1

        moment2 = paddle.to_tensor(torch_optim_state['optimizer']['state'][index]['exp_avg_sq'].numpy(), dtype=paddle.float32) if "embedding" in param.name or len(param.shape) == 1 else paddle.to_tensor(torch_optim_state['optimizer']['state'][index]['exp_avg_sq'].numpy().T, dtype=paddle.float32)
        assert moment2.shape == param.shape, "m2.shape : {}     param.shape : {}      param.name : {}".format(moment2.shape, param.shape, param.name)
        paddle_optim_state["{}_fp32_master_0_moment2_0".format(param.name)] = moment2

        beta1 = torch_optim_state['optimizer']['param_groups'][0]['betas'][0]
        beta2 = torch_optim_state['optimizer']['param_groups'][0]['betas'][1]
        step = torch_optim_state['optimizer']['param_groups'][0]['step']
        paddle_optim_state["{}_fp32_master_0_beta1_pow_acc_0".format(param.name)] = paddle.to_tensor([beta1 ** (step + 1)])
        paddle_optim_state["{}_fp32_master_0_beta2_pow_acc_0".format(param.name)] = paddle.to_tensor([beta2 ** (step + 1)])

        paddle_optim_state['LR_Scheduler'] = {
            'last_epoch': torch_state['opt_param_scheduler']['num_steps'],
            'last_lr': torch_optim_state['optimizer']['param_groups'][0]['lr'],
        }
    return paddle_optim_state

if __name__ == "__main__":
    import torch
    news = torch.load(
        '/root/paddlejob/workspace/env_run/gpt_benchmark/Megatron-LM/ckpt_345m_init.bin',
        map_location="cpu")
    paddle = trans_for_non_pp(news)
    pass

