import os
import pickle
import paddle
import numpy as np


def match_embedding_param(convert_parameter_name_dict, static_para_prefix=""):
    convert_parameter_name_dict[
        "embeddings.word_embeddings.weight"] = static_para_prefix + "word_embedding"
    convert_parameter_name_dict[
        "embeddings.position_embeddings.weight"] = static_para_prefix + "pos_embedding"
    convert_parameter_name_dict[
        "embeddings.token_type_embeddings.weight"] = static_para_prefix + "sent_embedding"
    convert_parameter_name_dict[
        "embeddings.task_type_embeddings.weight"] = static_para_prefix + "task_embedding"
    convert_parameter_name_dict[
        "embeddings.layer_norm.weight"] = static_para_prefix + "pre_encoder_layer_norm_scale"
    convert_parameter_name_dict[
        "embeddings.layer_norm.bias"] = static_para_prefix + "pre_encoder_layer_norm_bias"
    return convert_parameter_name_dict


def match_encoder_param(convert_parameter_name_dict,
                        layer_num=4,
                        static_para_prefix=""):
    dygraph_proj_names = ["q", "k", "v", "out"]
    static_proj_names = ["query", "key", "value", "output"]
    dygraph_param_names = ["weight", "bias"]
    static_param_names = ["w", "b"]
    dygraph_layer_norm_param_names = ["weight", "bias"]
    static_layer_norm_param_names = ["scale", "bias"]

    # Firstly, converts the multihead_attention to the parameter.
    dygraph_format_name = "encoder.layers.{}.self_attn.{}_proj.{}"
    static_format_name = static_para_prefix + "encoder_layer_{}_multi_head_att_{}_fc.{}_0"
    for i in range(0, layer_num):
        for dygraph_proj_name, static_proj_name in zip(dygraph_proj_names,
                                                       static_proj_names):
            for dygraph_param_name, static_param_name in zip(
                    dygraph_param_names, static_param_names):
                convert_parameter_name_dict[dygraph_format_name.format(i, dygraph_proj_name, dygraph_param_name)] = \
                    static_format_name.format(i, static_proj_name, static_param_name)

    # Secondly, converts the encoder ffn parameter.
    dygraph_ffn_linear_format_name = "encoder.layers.{}.linear{}.{}"
    static_ffn_linear_format_name = static_para_prefix + "encoder_layer_{}_ffn_fc_{}.{}_0"
    for i in range(0, layer_num):
        for j in range(0, 2):
            for dygraph_param_name, static_param_name in zip(
                    dygraph_param_names, static_param_names):
                convert_parameter_name_dict[dygraph_ffn_linear_format_name.format(i, j+1,  dygraph_param_name)] = \
                  static_ffn_linear_format_name.format(i, j, static_param_name)

    # Thirdly, converts the multi_head layer_norm parameter.
    dygraph_encoder_attention_layer_norm_format_name = "encoder.layers.{}.norm1.{}"
    static_encoder_attention_layer_norm_format_name = static_para_prefix + "encoder_layer_{}_post_att_layer_norm_{}"
    for i in range(0, layer_num):
        for dygraph_param_name, static_pararm_name in zip(
                dygraph_layer_norm_param_names, static_layer_norm_param_names):
            convert_parameter_name_dict[dygraph_encoder_attention_layer_norm_format_name.format(i, dygraph_param_name)] = \
                static_encoder_attention_layer_norm_format_name.format(i, static_pararm_name)

    dygraph_encoder_ffn_layer_norm_format_name = "encoder.layers.{}.norm2.{}"
    static_encoder_ffn_layer_norm_format_name = static_para_prefix + "encoder_layer_{}_post_ffn_layer_norm_{}"
    for i in range(0, layer_num):
        for dygraph_param_name, static_pararm_name in zip(
                dygraph_layer_norm_param_names, static_layer_norm_param_names):
            convert_parameter_name_dict[dygraph_encoder_ffn_layer_norm_format_name.format(i, dygraph_param_name)] = \
                 static_encoder_ffn_layer_norm_format_name.format(i, static_pararm_name)
    return convert_parameter_name_dict


def match_pooler_parameter(convert_parameter_name_dict, static_para_prefix=""):
    convert_parameter_name_dict[
        "pooler.dense.weight"] = static_para_prefix + "pooled_fc.w_0"
    convert_parameter_name_dict[
        "pooler.dense.bias"] = static_para_prefix + "pooled_fc.b_0"
    return convert_parameter_name_dict


def match_mlm_parameter(convert_parameter_name_dict, static_para_prefix=""):
    # convert_parameter_name_dict["cls.predictions.decoder_weight"] = "word_embedding"
    convert_parameter_name_dict[
        "cls.predictions.decoder_bias"] = static_para_prefix + "mask_lm_out_fc.b_0"
    convert_parameter_name_dict[
        "cls.predictions.transform.weight"] = static_para_prefix + "mask_lm_trans_fc.w_0"
    convert_parameter_name_dict[
        "cls.predictions.transform.bias"] = static_para_prefix + "mask_lm_trans_fc.b_0"
    convert_parameter_name_dict[
        "cls.predictions.layer_norm.weight"] = static_para_prefix + "mask_lm_trans_layer_norm_scale"
    convert_parameter_name_dict[
        "cls.predictions.layer_norm.bias"] = static_para_prefix + "mask_lm_trans_layer_norm_bias"
    return convert_parameter_name_dict


def match_last_fc_parameter(convert_parameter_name_dict, static_para_prefix=""):
    convert_parameter_name_dict["classifier.weight"] = "_cls_out_w"
    convert_parameter_name_dict["classifier.bias"] = "_cls_out_b"
    return convert_parameter_name_dict


def convert_static_to_dygraph_params(dygraph_params_save_path,
                                     static_params_dir,
                                     static_to_dygraph_param_name,
                                     model_name='static'):
    files = os.listdir(static_params_dir)

    state_dict = {}
    model_name = model_name
    for name in files:
        path = os.path.join(static_params_dir, name)
        # static_para_name = name.replace('@HUB_chinese-roberta-wwm-ext-large@',
        #                                 '')  # for hub module params
        static_para_name = name.replace('.npy', '')
        if static_para_name not in static_to_dygraph_param_name:
            print(static_para_name, "not in static_to_dygraph_param_name")
            continue
        dygraph_para_name = static_to_dygraph_param_name[static_para_name]
        value = paddle.load(path).numpy()
        if "cls" in dygraph_para_name or "classifier" in dygraph_para_name:
            # Note: cls.predictions parameters do not need add `model_name.` prefix
            state_dict[dygraph_para_name] = value
        else:
            state_dict[model_name + '.' + dygraph_para_name] = value

    with open(dygraph_params_save_path, 'wb') as f:
        pickle.dump(state_dict, f)
    params = paddle.load(dygraph_params_save_path)

    for name in state_dict.keys():
        if name in params:
            assert ((state_dict[name] == params[name].numpy()).all() == True)
        else:
            print(name, 'not in params')


if __name__ == "__main__":
    convert_parameter_name_dict = {}

    convert_parameter_name_dict = match_embedding_param(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_encoder_param(
        convert_parameter_name_dict, layer_num=12)
    convert_parameter_name_dict = match_pooler_parameter(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_mlm_parameter(
        convert_parameter_name_dict)

    static_to_dygraph_param_name = {
        value: key
        for key, value in convert_parameter_name_dict.items()
    }

    for static_name, dygraph_name in static_to_dygraph_param_name.items():
        print("{}:-------:{}".format(static_name, dygraph_name))

    convert_static_to_dygraph_params(
        dygraph_params_save_path='./dygraph_model/ernie_v1_chn_base.pdparams',
        static_params_dir='./ernie1.0_numpy/',
        static_to_dygraph_param_name=static_to_dygraph_param_name,
        model_name='ernie')
