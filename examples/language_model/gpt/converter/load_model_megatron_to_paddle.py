import sys
import paddle
import torch
import numpy as np

heads = 16
model = torch.load(sys.argv[1], map_location='cpu')

state = {}
for name, param in model["model"]["language_model"]["embedding"].items():
    for sub_name, sub_param in param.items():
        final_name = "{}.{}".format(name, sub_name)
        state[final_name] = sub_param.numpy()

for name, param in model["model"]["language_model"]["transformer"].items():
    final_name = "{}.{}".format('transformer', name)
    state[final_name] = param.numpy()

paddle.set_device("cpu")


def trans_name(key):
    k = key
    k = k.replace("transformer", "gpt.decoder")
    k = k.replace("mlp.dense_h_to_4h", "linear1")
    k = k.replace("mlp.dense_4h_to_h", "linear2")
    k = k.replace("attention.dense", "self_attn.out_proj")
    k = k.replace("input_layernorm", "norm1")
    k = k.replace("post_attention_layernorm", "norm2")
    k = k.replace("final_layernorm", "norm")
    k = k.replace("word_embeddings", "gpt.embeddings.word_embeddings")
    k = k.replace("position_embeddings", "gpt.embeddings.position_embeddings")
    return k


new_state_dict = {}
all_num = 0
for key in sorted(list(state.keys())):
    all_num += state[key].size
    new_key = trans_name(key)
    if "query_key_value" in key:
        shape = state[key].shape
        print(shape)
        if "weight" in key:
            # state[key] = state[key].transpose().reshape((shape[1], heads, shape[0]//heads))
            q, k, v = np.split(state[key], 3, axis=0)
            q = q.transpose()
            k = k.transpose()
            v = v.transpose()
            # q, k, v = np.split(state[key], 3, axis=-1)
            # q = q.reshape((shape[1], -1))
            # k = k.reshape((shape[1], -1))
            # v = v.reshape((shape[1], -1))
            pass
        else:
            #state[key] = state[key].transpose().reshape(heads, shape[0]//heads)
            print("BIAS SHAPE", state[key].shape, state[key].transpose().shape)
            #state[key] = state[key].transpose().reshape(heads, shape[0]//heads)
            q, k, v = np.split(state[key], 3, axis=-1)
            q = q.reshape((-1))
            k = k.reshape((-1))
            v = v.reshape((-1))
        q_name = new_key.replace("attention.query_key_value",
                                 "self_attn.q_proj")
        k_name = new_key.replace("attention.query_key_value",
                                 "self_attn.k_proj")
        v_name = new_key.replace("attention.query_key_value",
                                 "self_attn.v_proj")
        new_state_dict[q_name] = paddle.to_tensor(q, dtype="float32")
        new_state_dict[k_name] = paddle.to_tensor(k, dtype="float32")
        new_state_dict[v_name] = paddle.to_tensor(v, dtype="float32")
        continue
    for name in [
            "mlp.dense_h_to_4h.weight", "mlp.dense_4h_to_h.weight",
            "attention.dense.weight"
    ]:
        if name in key:
            state[key] = state[key].transpose()
    new_state_dict[new_key] = paddle.to_tensor(state[key], dtype="float32")
print("all shape numel:{}".format(all_num))
for key, value in new_state_dict.items():
    print("key:{}, shape:{}, dtype:{}".format(key, value.shape, value.dtype))
#paddle.save(new_state_dict, './gpt-medium-en.pdparams')
