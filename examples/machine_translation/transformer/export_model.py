import os
import yaml
import argparse
from pprint import pprint
from attrdict import AttrDict

import paddle

import reader

from paddlenlp.transformers import InferTransformerModel, position_encoding_init
from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/transformer.big.yaml",
        type=str,
        help="Path of the config file. ")
    args = parser.parse_args()
    return args


def do_export(args):
    # Adapt vocabulary size
    reader.adapt_vocab_size(args)
    # Define model
    transformer = InferTransformerModel(
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
        max_out_len=args.max_out_len)

    # Load the trained model
    assert args.init_from_params, (
        "Please set init_from_params to load the infer model.")

    model_dict = paddle.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))

    # To avoid a longer length than training, reset the size of position
    # encoding to max_length
    model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    transformer.load_dict(model_dict)
    # Set evaluate mode
    transformer.eval()

    # Convert dygraph model to static graph model 
    transformer = paddle.jit.to_static(
        transformer,
        input_spec=[
            # src_word
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")
        ])

    # Save converted static graph model
    paddle.jit.save(transformer,
                    os.path.join(args.inference_model_dir, "transformer"))
    logger.info("Transformer has been saved to {}".format(
        args.inference_model_dir))


if __name__ == "__main__":
    ARGS = parse_args()
    yaml_file = ARGS.config
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)

    do_export(args)
