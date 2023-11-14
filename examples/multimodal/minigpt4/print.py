import paddle
import numpy as np
vit_model = np.load("vit1.npy", allow_pickle=True).item()

# import pdb;pdb.set_trace()

paddle_model = {}

def trans(a):
    if len(a.shape) == 2:
        return paddle.to_tensor(a).transpose([1, 0])
    else:
        return paddle.to_tensor(a)

# --------------  resampler model --------------- #
for k, v in vit_model.items():
    if k.startswith("resampler.layers"):
        paddle_model[k] = trans(vit_model[k])
    elif k.startswith("resampler."):
        paddle_model[k] = paddle.to_tensor(vit_model[k])


paddle_model["opt_proj.weight"] = paddle.to_tensor(vit_model["opt_proj.weight"]).transpose([1, 0])
paddle_model["opt_proj.bias"] = paddle.to_tensor(vit_model["opt_proj.bias"])


# --------------  vit model --------------- #
paddle_model["vision_model.embeddings.class_embedding"] = paddle.to_tensor(vit_model["visual_encoder.class_embedding"]).reshape([1, 1, 1280])
paddle_model["vision_model.embeddings.position_embedding"] = paddle.to_tensor(vit_model["visual_encoder.positional_embedding"]).reshape([1, 257, 1280])
paddle_model["vision_model.embeddings.patch_embedding.weight"] = paddle.to_tensor(vit_model["visual_encoder.conv1.weight"])

paddle_model["vision_model.embeddings.ln_pre.weight"] = paddle.to_tensor(vit_model["visual_encoder.ln_pre.weight"])
paddle_model["vision_model.embeddings.ln_pre.bias"] = paddle.to_tensor(vit_model["visual_encoder.ln_pre.bias"])

paddle_model["vision_model.post_layernorm.weight"] = paddle.to_tensor(vit_model["ln_vision.weight"])
paddle_model["vision_model.post_layernorm.bias"] = paddle.to_tensor(vit_model["ln_vision.bias"])


for idx in range(32):
    paddle_model["vision_model.encoder.layers.{}.self_attn.qkv.weight".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.attn.in_proj_weight".format(idx)]).transpose([1, 0])
    paddle_model["vision_model.encoder.layers.{}.self_attn.qkv.bias".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.attn.in_proj_bias".format(idx)])
    paddle_model["vision_model.encoder.layers.{}.self_attn.projection.weight".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.attn.out_proj.weight".format(idx)]).transpose([1, 0])
    paddle_model["vision_model.encoder.layers.{}.self_attn.projection.bias".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.attn.out_proj.bias".format(idx)])

    paddle_model["vision_model.encoder.layers.{}.layer_norm1.weight".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.ln_1.weight".format(idx)])
    paddle_model["vision_model.encoder.layers.{}.layer_norm1.bias".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.ln_1.bias".format(idx)])

    paddle_model["vision_model.encoder.layers.{}.mlp.fc1.weight".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.mlp.c_fc.weight".format(idx)]).transpose([1, 0])
    paddle_model["vision_model.encoder.layers.{}.mlp.fc1.bias".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.mlp.c_fc.bias".format(idx)])
    paddle_model["vision_model.encoder.layers.{}.mlp.fc2.weight".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.mlp.c_proj.weight".format(idx)]).transpose([1, 0])
    paddle_model["vision_model.encoder.layers.{}.mlp.fc2.bias".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.mlp.c_proj.bias".format(idx)])

    paddle_model["vision_model.encoder.layers.{}.layer_norm2.weight".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.ln_2.weight".format(idx)])
    paddle_model["vision_model.encoder.layers.{}.layer_norm2.bias".format(idx)] = paddle.to_tensor(vit_model["visual_encoder.transformer.resblocks.{}.ln_2.bias".format(idx)])



# # ------------ gpt model -------------#

paddle_model["language_model.llama.embed_tokens.weight"] = paddle.to_tensor(vit_model["opt_model.model.embed_tokens.weight"])
paddle_model["language_model.llama.norm.weight"] = paddle.to_tensor(vit_model["opt_model.model.norm.weight"])
paddle_model["language_model.lm_head.weight"] = paddle.to_tensor(vit_model["opt_model.lm_head.weight"]).transpose([1,0])


for idx in range(32):
    paddle_model["language_model.llama.layers.{}.self_attn.qkv_proj.weight".format(idx)] = paddle.to_tensor(vit_model["opt_model.model.layers.{}.self_attn.W_pack.weight".format(idx)]).transpose([1,0])
    paddle_model["language_model.llama.layers.{}.self_attn.o_proj.weight".format(idx)] = paddle.to_tensor(vit_model["opt_model.model.layers.{}.self_attn.o_proj.weight".format(idx)]).transpose([1,0])
    # paddle_model["language_model.llama.layers.{}.self_attn.rotary_emb.inv_freq".format(idx)] = paddle.to_tensor(vit_model["opt_model.model.layers.{}.self_attn.rotary_emb.inv_freq".format(idx)])
    paddle_model["language_model.llama.layers.{}.mlp.gate_proj.weight".format(idx)] = paddle.to_tensor(vit_model["opt_model.model.layers.{}.mlp.gate_proj.weight".format(idx)]).transpose([1,0])
    paddle_model["language_model.llama.layers.{}.mlp.down_proj.weight".format(idx)] = paddle.to_tensor(vit_model["opt_model.model.layers.{}.mlp.down_proj.weight".format(idx)]).transpose([1,0])
    paddle_model["language_model.llama.layers.{}.mlp.up_proj.weight".format(idx)] = paddle.to_tensor(vit_model["opt_model.model.layers.{}.mlp.up_proj.weight".format(idx)]).transpose([1,0])
    
    
    paddle_model["language_model.llama.layers.{}.input_layernorm.weight".format(idx)] = paddle.to_tensor(vit_model["opt_model.model.layers.{}.input_layernorm.weight".format(idx)])
    paddle_model["language_model.llama.layers.{}.post_attention_layernorm.weight".format(idx)] = paddle.to_tensor(vit_model["opt_model.model.layers.{}.post_attention_layernorm.weight".format(idx)])



paddle.save(paddle_model, "/root/paddlejob/workspace/env_run/zhengshifeng/vitllm/pr/PaddleNLP/examples/multimodal/minigpt4/vit_model/model_state.pdparams")
