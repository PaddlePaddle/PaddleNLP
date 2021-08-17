import math
import numpy as np
import torch
import tensorflow as tf


def _check_rule(tensor_name, rule):
    if "Adam" in tensor_name or "adam" in tensor_name:
        return False
    assert isinstance(rule, str) and rule
    rule = rule.split("-")
    assert len(rule) < 3
    if len(rule) == 2:
        white, black = rule[0].split(" "), rule[1].split(" ")
    else:
        white, black = rule[0].split(" "), []
    for b in black:
        if b in tensor_name.split("."):
            return False
    for w in white:
        if w not in tensor_name.split("."):
            return False
    return True


def _apply_rule(proto_name, ckpt_rule, tensor_names, state_dict):
    expression = [
        ele for ele in ckpt_rule.split("&&") if ele.startswith("expression_")
    ]

    ckpt_rule = [
        ele for ele in ckpt_rule.split("&&")
        if not ele.startswith("expression_")
    ]

    assert (len(ckpt_rule) > 0 and len(expression) < 2) or (
        len(ckpt_rule) == 0 and len(expression) > 0)

    if len(expression) < 2:
        expression = "" if not expression else expression[0].split("_")[1]
    else:
        expression = [exp.split("_")[1] for exp in expression]

    target_tn = []
    for cr in ckpt_rule:
        tmp = []
        for tn in tensor_names:
            if _check_rule(tn, cr):
                tmp.append(tn)
        if len(tmp) != 1:
            print(tmp, cr)
        assert len(tmp) == 1
        target_tn.extend(tmp)
    target_tensor = [state_dict[name] for name in target_tn]
    tt = {}
    if target_tensor:
        exec("tt['save'] = [ele%s for ele in target_tensor]" % expression)
    else:
        if not isinstance(expression, list):
            expression = [expression]
        exec("tt['save'] = [%s]" % ",".join(expression))

    target_tensor = np.concatenate(tt["save"], axis=-1)
    print("%s -> %s, shape: %s, convert finished." %
          (target_tn
           if target_tn else "created", proto_name, target_tensor.shape))
    return target_tensor


def fill_proto_layer(tensor_names, state_dict, layer, mapping_dict):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = _apply_rule(proto_name, ckpt_rule, tensor_names,
                                    state_dict)
        exec("layer.%s[:]=target_tensor.flatten().tolist()" % proto_name)


def fill_hdf5_layer(tensor_names, state_dict, hdf5_file, hdf5_dataset_prefix,
                    mapping_dict):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = _apply_rule(proto_name, ckpt_rule, tensor_names,
                                    state_dict)
        hdf5_file.create_dataset(
            hdf5_dataset_prefix + proto_name,
            data=target_tensor.flatten().tolist())


def _get_encode_output_mapping_dict(dec_layer_num):
    encode_output_kernel_pattern = [
        "encoder_attn {0} k_proj weight&&encoder_attn {0} v_proj weight".format(
            ele) for ele in range(dec_layer_num)
    ]
    encode_output_bias_pattern = [
        "encoder_attn {0} k_proj bias&&encoder_attn {0} v_proj bias".format(ele)
        for ele in range(dec_layer_num)
    ]

    return {
        "encode_output_project_kernel_kv": "&&".join(
            encode_output_kernel_pattern + ["expression_.transpose(0, 1)"]),
        "encode_output_project_bias_kv": "&&".join(encode_output_bias_pattern),
    }


def _get_position_encoding(length,
                           hidden_size,
                           min_timescale=1.0,
                           max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    with tf.device("/cpu:0"):
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = hidden_size // 2
        log_timescale_increment = math.log(
            float(max_timescale) /
            float(min_timescale)) / (tf.cast(num_timescales, tf.float32) - 1)
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) *
            -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat(
            [tf.math.sin(scaled_time), tf.math.cos(scaled_time)], axis=1)
    return signal


def _gather_token_embedding(tensor_names, name2var_dict, tn_pattern, lang="en"):
    """use pattern to diff source and target."""
    target_tn = []
    # lang_embedding = None
    for tn in tensor_names:
        if (tn_pattern in tn.split(".")) and ("weight" in tn.split(".")):
            target_tn.append(tn)
            continue
        # if tn == "lang_embeddings.weight":
        #     lang_embedding = name2var_dict[tn].numpy()
        # target_tn = sorted(target_tn, key=lambda x: int(x.split('_')[-1]))
        # print(target_tn)
    target_tensor = [name2var_dict[name] for name in target_tn]
    target_tensor = np.concatenate(target_tensor, axis=0)
    # target_tensor = target_tensor * (target_tensor.shape[1] ** 0.5)
    # print(
    #     "lang embedding shape: {}, added {} embedding to token embeddings".format(
    #         lang, lang_embedding.shape
    #     )
    # )
    # target_tensor += lang_embedding[LANG2ID[lang]]
    print("token embedding shape is {}".format(target_tensor.shape))
    # print("token embedding shape is %s" % target_tensor.shape)

    return target_tensor
