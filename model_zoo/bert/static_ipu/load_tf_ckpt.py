# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


def get_tf_mapping(args):
    squad_mapping = {
        "cls/squad/output_weights": "linear_72.w_0",
        "cls/squad/output_bias": "linear_72.b_0"
    }

    tf_to_pdmodel = {
        "bert/embeddings/word_embeddings": "ipu_bert_embeddings_0.w_0",
        "bert/embeddings/position_embeddings": "embedding_0.w_0",
        "bert/embeddings/token_type_embeddings": "ipu_bert_embeddings_0.w_1",
        "bert/embeddings/LayerNorm/gamma": "layer_norm_0.w_0",
        "bert/embeddings/LayerNorm/beta": "layer_norm_0.b_0"
    }
    for i in range(args.num_hidden_layers):
        layer = {
            f"bert/encoder/layer_{i}/attention/self/query/bias":
            f"bert_model_0.b_{i}",
            f"bert/encoder/layer_{i}/attention/self/key/bias":
            f"bert_model_0.b_{i}",
            f"bert/encoder/layer_{i}/attention/self/value/bias":
            f"bert_model_0.b_{i}",
            f"bert/encoder/layer_{i}/attention/output/dense/kernel":
            f"linear_{i*6}.w_0",
            f"bert/encoder/layer_{i}/attention/output/dense/bias":
            f"linear_{i*6}.b_0",
            f"bert/encoder/layer_{i}/attention/output/LayerNorm/gamma":
            f"layer_norm_{i*4+2}.w_0",
            f"bert/encoder/layer_{i}/attention/output/LayerNorm/beta":
            f"layer_norm_{i*4+2}.b_0",
            f"bert/encoder/layer_{i}/intermediate/dense/kernel":
            f"linear_{i*6+2}.w_0",
            f"bert/encoder/layer_{i}/intermediate/dense/bias":
            f"linear_{i*6+2}.b_0",
            f"bert/encoder/layer_{i}/output/dense/kernel":
            f"linear_{i*6+3}.w_0",
            f"bert/encoder/layer_{i}/output/dense/bias":
            f"linear_{i*6+3}.b_0",
            f"bert/encoder/layer_{i}/output/LayerNorm/gamma":
            f"layer_norm_{(i+1)*4}.w_0",
            f"bert/encoder/layer_{i}/output/LayerNorm/beta":
            f"layer_norm_{(i+1)*4}.b_0",
        }
        layer[
            f"bert/encoder/layer_{i}/attention/self/query/kernel"] = f"bert_model_0.w_{i*3+0}"
        layer[
            f"bert/encoder/layer_{i}/attention/self/key/kernel"] = f"bert_model_0.w_{i*3+1}"
        layer[
            f"bert/encoder/layer_{i}/attention/self/value/kernel"] = f"bert_model_0.w_{i*3+2}"
        tf_to_pdmodel.update(**layer)

    if args.task == "PRETRAINING":
        logger.error("Mapping ckpt weights is only supported in SQUAD task.")
    elif args.task == "SQUAD":
        tf_to_pdmodel.update(**squad_mapping)

    return tf_to_pdmodel


def generate_initializers(args, map_names, load_data, mapping, transform={}):
    initializers = {}
    initializers_param = {}
    initializers_opt = {}

    qkv_tensor_range = {
        "query": (0, args.hidden_size),
        "key": (args.hidden_size, args.hidden_size * 2),
        "value": (args.hidden_size * 2, args.hidden_size * 3),
    }

    for name, array in zip(map_names, load_data):
        logger.debug(
            f"Initialising tensor from checkpoint {name} -> {mapping[name]}")

        # config["lamb_m_dtype"] is for setting the data type for accl1 of lamb
        # BERT can use FP16 for accl1 without lossing accuracy
        # accl2 is always in FP32
        lamb_m_dtype = np.float32
        dtype = np.float32

        if "moment1" in mapping[name]:
            if array.dtype != lamb_m_dtype:
                array = array.astype(lamb_m_dtype)
        elif "moment2" in mapping[name]:
            if array.dtype != np.float32:
                array = array.astype(np.float32)
        elif array.dtype != dtype:
            array = array.astype(dtype)

        # If it's part of QKV biases, we need to handle separately as those 3
        # tensors need concatenating into one
        if "bert_model_0.b" in mapping[name]:
            qkv_part = name.split("/")[5]
            if mapping[name] not in initializers.keys():
                qkv_shape = (array.shape[0] * 3)
                initializers[mapping[name]] = np.empty(qkv_shape,
                                                       dtype=array.dtype)

            start_idx = qkv_tensor_range[qkv_part][0]
            end_idx = qkv_tensor_range[qkv_part][1]
            initializers[mapping[name]][start_idx:end_idx] = array
            logger.debug(
                f"Initialising QKV_bias component {name}[{start_idx}:{end_idx}] from checkpoint"
            )
            continue

        if name in transform:
            array = transform[name](array)

        padded_vocab_length = args.vocab_size
        if "bert_embeddings_0.w_0" in mapping[name]:
            tf_vocab_length = array.shape[0]
            diff = padded_vocab_length - tf_vocab_length
            # Pad or Crop the vocab.
            if diff > 0:
                logger.info(
                    f"Padding the vocabulary. From {tf_vocab_length} to {padded_vocab_length}"
                )
                pad = np.zeros((diff, args.hidden_size)).astype(array.dtype)
                array = np.concatenate((array, pad), axis=0)
            else:
                logger.warning(
                    f"Cropping the vocabulary may negatively effect performance. From {tf_vocab_length} to {padded_vocab_length}"
                )
                array = np.array(array[:padded_vocab_length, :])
            # if args.task == "PRETRAINING":
            # We use transposed weight in both pretraining and squad
            array = np.transpose(array, [1, 0])

        if "embedding_0.w_0" in mapping[name]:
            max_pos, hidden_len = array.shape
            if max_pos > args.max_position_embeddings:
                array = array[:args.max_position_embeddings, :]

            # Otherwise just copy the positional embeddings over and over again as is done in longformer
            elif max_pos < args.max_position_embeddings:
                logger.warning(
                    f"Not enough positional embeddings in checkpoint, copying to match length..."
                )
                array = array[np.mod(np.arange(args.max_position_embeddings),
                                     max_pos)]

        initializers[mapping[name]] = array.copy()
        for k in initializers:
            if "moment" in k:
                initializers_opt[k] = initializers[k]
            else:
                initializers_param[k] = initializers[k]
    return initializers_param, initializers_opt


# util function for load tf pretrained weight
def load_initializers_from_tf(file_path, args):
    """
    Loads weights, etc. from Tensorflow files into a dictionary of Numpy Arrays.

    Can read either checkpoint files, or frozen graphs, according to the
    `is_checkpoint` flag, passed in as the second argument.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model requires TensorFlow to be installed. "
            "Please see https://www.tensorflow.org/install/ for installation "
            "instructions.")
        raise

    tf_path = os.path.abspath(file_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)

    mapping = get_tf_mapping(args)
    map_names = [name for name, shape in init_vars if name in mapping.keys()]
    for name in (n for n, _ in init_vars if n not in mapping.keys()):
        logger.debug(f"Skipping load of {name} - Not in mapping")

    load_data = [tf.train.load_variable(tf_path, name) for name in map_names]
    initializers, opt_params = generate_initializers(args, map_names, load_data,
                                                     mapping)
    return initializers, opt_params
