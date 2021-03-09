import sys
import os
import numpy as np
from attrdict import AttrDict
import argparse
import time

import paddle

import yaml
from pprint import pprint

from paddlenlp.ext_op import FasterTransformer

from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./sample/config/decoding.sample.yaml",
        type=str,
        help="Path of the config file. ")
    parser.add_argument(
        "--decoding-lib",
        default="./build/lib/libdecoding_op.so",
        type=str,
        help="Path of libdecoding_op.so. ")
    parser.add_argument(
        "--use-fp16-decoding",
        action="store_true",
        help="Whether to use fp16 decoding to predict. ")
    args = parser.parse_args()
    return args


def generate_encoder_result(batch_size, max_seq_len, memory_hidden_dim, dtype):
    memory_sequence_length = np.random.randint(
        1, max_seq_len, size=batch_size).astype(np.int32)
    memory_sequence_length[np.random.randint(0, batch_size)] = max_seq_len
    outter_embbeding = np.random.randn(memory_hidden_dim) * 0.01

    memory = []
    mem_max_seq_len = np.max(memory_sequence_length)
    for i in range(batch_size):
        data = np.random.randn(mem_max_seq_len, memory_hidden_dim) * 0.01
        for j in range(memory_sequence_length[i], mem_max_seq_len):
            data[j] = outter_embbeding
        memory.append(data)
    memory = np.asarray(memory)
    memory = paddle.to_tensor(memory, dtype=dtype)
    memory_sequence_length = paddle.to_tensor(
        memory_sequence_length, dtype="int32")

    return memory, memory_sequence_length


def do_predict(args):
    place = "gpu"
    place = paddle.set_device(place)

    # Define model
    transformer = FasterTransformer(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        beam_size=args.beam_size,
        max_out_len=args.max_out_len,
        decoding_lib=args.decoding_lib,
        use_fp16_decoding=args.use_fp16_decoding)

    # Set evaluate mode
    transformer.eval()

    enc_output, mem_seq_len = generate_encoder_result(
        args.infer_batch_size, args.max_length, args.d_model, "float16"
        if args.use_fp16_decoding else "float32")
    with paddle.no_grad():
        for i in range(100):
            # For warmup. 
            if 50 == i:
                start = time.time()
            transformer.decoding(
                enc_output=enc_output, memory_seq_lens=mem_seq_len)
        logger.info("Average test time for decoding is %f ms" % (
            (time.time() - start) / 50 * 1000))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
    args.decoding_lib = ARGS.decoding_lib
    args.use_fp16_decoding = ARGS.use_fp16_decoding

    do_predict(args)
