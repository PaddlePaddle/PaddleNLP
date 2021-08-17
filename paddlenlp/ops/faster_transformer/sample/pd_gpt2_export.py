import os
import h5py
import numpy as np
from collections import OrderedDict
from transformers import GPT2LMHeadModel
from paddlenlp.transformers import GPTLMHeadModel
import paddle
from utils import fill_hdf5_layer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""
For the mapping dictionary: key is the value of the proto parameter,
value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_.
You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""

enc_layer_mapping_dict_paddle = OrderedDict({
    "multihead_norm_scale": "norm1 weight",
    "multihead_norm_bias": "norm1 bias",
    # GPT2's Conv1D don't need transpose
    # https://github.com/huggingface/transformers/blob/9ec0f01b6c3aff4636869aee735859fb6f89aa98/src/transformers/modeling_utils.py#L1400
    "multihead_project_kernel_qkv": "self_attn c_attn weight",
    "multihead_project_bias_qkv": "self_attn c_attn bias",
    "multihead_project_kernel_output": "self_attn out_proj weight",
    "multihead_project_bias_output": "self_attn out_proj bias",
    "ffn_norm_scale": "norm2 weight",
    "ffn_norm_bias": "norm2 bias",
    "ffn_first_kernel": "linear1 weight",
    "ffn_first_bias": "linear1 bias",
    "ffn_second_kernel": "linear2 weight",
    "ffn_second_bias": "linear2 bias",
})

src_emb_mapping_dict_paddle = OrderedDict({
    "norm_scale": "norm weight",
    "norm_bias": "norm bias",
    "token_embedding": "word_embeddings",
    # manually process position_embedding to customize for max_step
    # "position_embedding": "wpe",
})


def extract_paddle_gpt_weights(
        output_file,
        model_dir,
        head_num,
        generation_method,
        topk=1,
        topp=0.75,
        eos_id=7,
        pad_id=0,
        max_step=50, ):
    # load var names
    encoder_state_dict = GPTLMHeadModel.from_pretrained(model_dir).state_dict()
    enc_var_name_list = list(encoder_state_dict.keys())

    # initialize output file
    output_file += ".hdf5"
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))
    hdf5_file = h5py.File(output_file, "w")

    # fill each encoder layer's params
    enc_tensor_names = {}
    for name in enc_var_name_list:
        name_split = name.split(".")

        if len(name_split) <= 2 or not name_split[3].isdigit():
            continue
        layer_id = int(name_split[3])
        if name_split[-2] in ['q_proj', 'k_proj', 'v_proj']:
            name_split[-2] = 'c_attn'
            new_name = '.'.join(name_split)
            if layer_id not in enc_tensor_names.keys(
            ) or new_name not in enc_tensor_names[layer_id]:
                enc_tensor_names.setdefault(layer_id, []).append(new_name)
        else:
            enc_tensor_names.setdefault(layer_id, []).append(name)

    # fill encoder_stack
    for layer_id in sorted(enc_tensor_names.keys()):
        encoder_state_dict[str(layer_id).join([
            'gpt.decoder.layers.', '.self_attn.c_attn.weight'
        ])] = paddle.concat(
            [
                encoder_state_dict[str(layer_id).join(
                    ['gpt.decoder.layers.', '.self_attn.q_proj.weight'])],
                encoder_state_dict[str(layer_id).join(
                    ['gpt.decoder.layers.', '.self_attn.k_proj.weight'])],
                encoder_state_dict[str(layer_id).join(
                    ['gpt.decoder.layers.', '.self_attn.v_proj.weight'])]
            ],
            axis=-1)

        encoder_state_dict[str(layer_id).join(
            ['gpt.decoder.layers.', '.self_attn.c_attn.bias'])] = paddle.concat(
                [
                    encoder_state_dict[str(layer_id).join(
                        ['gpt.decoder.layers.', '.self_attn.q_proj.bias'])],
                    encoder_state_dict[str(layer_id).join(
                        ['gpt.decoder.layers.', '.self_attn.k_proj.bias'])],
                    encoder_state_dict[str(layer_id).join(
                        ['gpt.decoder.layers.', '.self_attn.v_proj.bias'])]
                ],
                axis=-1)
        fill_hdf5_layer(
            enc_tensor_names[layer_id],
            encoder_state_dict,
            hdf5_file,
            f"encoder_stack/{layer_id}/",
            enc_layer_mapping_dict_paddle, )

    # fill src_embedding - except for position embedding
    fill_hdf5_layer(
        enc_var_name_list,
        encoder_state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict_paddle, )

    # special handling for position embedding
    position_emb = encoder_state_dict[
        "gpt.embeddings.position_embeddings.weight"]
    _max_allowed_step, _hidden_size = position_emb.shape
    if max_step > _max_allowed_step:
        print(f"max_step {max_step} exceed max allowed step, abort.")
        return
    # truncate position embedding for max_step
    position_emb = position_emb[:max_step, :]
    print(
        f"processed position_embedding with max_step constriant, shape: {position_emb.shape}"
    )
    position_emb = position_emb.flatten().tolist()
    hdf5_file.create_dataset(
        "src_embedding/position_embedding", data=position_emb, dtype="f4")

    # save number of layers metadata
    hdf5_file.create_dataset(
        "model_conf/n_encoder_stack", data=len(enc_tensor_names), dtype="i4")
    # fill in model_conf
    hdf5_file.create_dataset("model_conf/head_num", data=head_num, dtype="i4")
    hdf5_file.create_dataset(
        "model_conf/src_padding_id", data=pad_id, dtype="i4")
    hdf5_file.create_dataset(
        "model_conf/sampling_method",
        data=np.array([ord(c) for c in generation_method]).astype(np.int8),
        dtype="i1", )
    hdf5_file.create_dataset("model_conf/topp", data=topp, dtype="f4")
    hdf5_file.create_dataset("model_conf/topk", data=topk, dtype="i4")
    hdf5_file.create_dataset("model_conf/eos_id", data=eos_id, dtype="i4")

    hdf5_file.close()
    # read-in again to double check
    hdf5_file = h5py.File(output_file, "r")

    def _print_pair(key, value):
        if key == "sampling_method":
            value = "".join(map(chr, value[()]))
        else:
            value = value[()]
        print(f"{key}: {value}")

    list(map(lambda x: _print_pair(*x), hdf5_file["model_conf"].items()))


if __name__ == "__main__":
    output_lightseq_model_name = "lightseq_gpt2_base"
    output_lightseq_model_name_paddle = "lightseq_gpt2_base_paddle"  # or "lightseq_gpt2_large"
    input_huggingface_gpt_model = "gpt2"
    input_paddlenlp_gpt_model = "gpt2-en"  # or "gpt2-large"
    head_number = 12  # 20 for "gpt2-large"
    # generation_method should be "topk" or "topp"
    generation_method = "topk"
    topk = 1
    topp = 0.75
    max_step = 50
    eos_id = 50256
    pad_id = 50257
    extract_paddle_gpt_weights(
        output_lightseq_model_name_paddle,
        input_paddlenlp_gpt_model,
        head_num=head_number,  # layer number
        generation_method=generation_method,
        topk=topk,
        topp=topp,
        eos_id=eos_id,
        pad_id=pad_id,
        max_step=max_step, )
