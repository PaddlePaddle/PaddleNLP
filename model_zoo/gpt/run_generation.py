import random
import argparse

import numpy as np
import paddle
from paddlenlp.transformers import (
    GPTLMHeadModel,
    GPTForGeneration,
    GPTTokenizer,
    GPTChineseTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPTLMHeadModel, GPTTokenizer),
    # "gpt2-cn": (GPTLMHeadModel, GPTChineseTokenizer),
    "gpt2-cn": (GPTForGeneration, GPTChineseTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_type',
                        default='gpt2-cn',
                        type=str,
                        help="Model type selected in the list: " +
                        ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument(
        '--model_name_or_path',
        default='gpt-cpm-small-cn-distill',
        type=str,
        help="The path or shortcut name of the pre-trained model.")
    parser.add_argument('--decode_strategy',
                        type=str,
                        default='greedy_search',
                        help='The decode strategy in generation.')
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help=
        'The number of highest probability vocabulary tokens to keep for top-k sampling.'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p',
                        type=float,
                        default=1.0,
                        help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams',
                        type=int,
                        default=0,
                        help='The number of beams for beam search.')
    parser.add_argument(
        '--length_penalty',
        type=float,
        default=1.0,
        help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument(
        '--early_stopping',
        type=eval,
        default=False,
        help=
        'Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.'
    )
    parser.add_argument('--min_dec_len',
                        type=int,
                        default=1,
                        help='The minimum sequence length of generation.')
    parser.add_argument('--max_dec_len',
                        type=int,
                        default=16,
                        help='The maximum sequence length of generation.')
    parser.add_argument('--num_return_sequences',
                        type=int,
                        default=1,
                        help='The number of output sequences to generation.')
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        help='Random seed for initialization.')
    parser.add_argument('--device',
                        type=str,
                        default='gpu',
                        help='The device to select for training the model.')

    args = parser.parse_args()
    return args


def print_args(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 or length > max_sequence_length:
        length = max_sequence_length
    return length


def left_padding(inputs, pad_id, padding="longest"):
    assert "input_ids" in inputs, "input_ids should be in inputs!"
    max_length = 0
    for ids in inputs["input_ids"]:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(name, max_length, to_pad_id):
        values = inputs[name]
        res = []
        for index, value in enumerate(values):
            res.append(extend_max_lenth(value, max_length, to_pad_id))
        inputs[name] = res

    extend_filed("input_ids", max_length, pad_id)
    if "attention_mask" in inputs:
        extend_filed("attention_mask", max_length, 0)
    if "position_ids" in inputs:
        extend_filed("position_ids", max_length, 0)

    return inputs


def main(args, input_text):
    paddle.set_device(args.device)
    if args.seed:
        set_seed(args.seed)

    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(
            "The `model_type` must be selected in the list: {}. But received: {}."
            .format(MODEL_CLASSES.keys(), args.model_type))

    model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model.eval()

    args.max_dec_len = adjust_length_to_model(args.max_dec_len, 512)
    #                                           model.max_position_embeddings)

    print(input_text)
    # input_ids = tokenizer.encode(input_text)['input_ids']
    inputs = tokenizer(input_text)
    inputs = left_padding(inputs, tokenizer.bos_token_id)
    input_ids = inputs['input_ids']
    print(input_ids)
    if len(input_ids) == 0:
        input_ids = None
    else:
        # [1, seq_len]
        input_ids = paddle.to_tensor(input_ids, dtype='int64')  # .unsqueeze(0)
        print(input_ids.shape)

    ids, scores = model(input_ids=input_ids,
                        max_length=args.max_dec_len,
                        min_length=args.min_dec_len,
                        eos_token_id=tokenizer.eos_token_id,
                        decode_strategy=args.decode_strategy,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        early_stopping=args.early_stopping,
                        num_return_sequences=args.num_return_sequences)

    generated_sequences = []
    print(ids)
    for i, generated_ids in enumerate(ids):
        print("*" * 10 + " GENERATED SEQUENCE {} ".format(i) + "*" * 10)
        generated_ids = generated_ids.numpy().tolist()
        # Decode text
        text = tokenizer.convert_ids_to_string(generated_ids)
        # Add the prompt at the beginning of the sequence.
        sequence = input_text[i] + text
        generated_sequences.append(sequence)
        print(sequence)

    return generated_sequences


if __name__ == "__main__":
    args = parse_args()
    input_text = ['花间一壶酒，独酌无相亲。举杯邀明月，', '我寄愁心与明月，随风']
    # input_text = ['我寄愁心与明月，', '花间一壶酒，独酌无相亲。举杯邀明月，']
    # input_text = ['花间一壶酒，独酌无相亲。举杯', '花间一壶酒，独酌无相亲。举杯邀明月，']
    # input_text = ['花间一壶酒，独酌无相亲。举杯', '花间一壶酒，独酌无相亲。举杯']
    input_text = ['床前明月光，疑是地上霜。举头望明月，低头', '白日依山尽，黄河入海流。欲穷千里目，']
    main(args, input_text)
