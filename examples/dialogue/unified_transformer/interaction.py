import argparse

from termcolor import colored, cprint
import paddle
from paddlenlp.transformers import (UnifiedTransformerLMHeadModel,
                                    UnifiedTransformerTokenizer)

from utils import print_args, set_seed, select_response


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_name_or_path', type=str, default='plato-mini', help='The path or shortcut name of the pre-trained model.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for initialization.')
    parser.add_argument('--min_dec_len', type=int, default=1, help='The minimum sequence length of generation.')
    parser.add_argument('--max_dec_len', type=int, default=64, help='The maximum sequence length of generation.')
    parser.add_argument('--num_return_sequences', type=int, default=20, help='The numbers of returned sequences for one input in generation.')
    parser.add_argument('--decode_strategy', type=str, default='sampling', help='The decode strategy in generation.')
    parser.add_argument('--top_k', type=int, default=5, help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p', type=float, default=1.0, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', type=int, default=0, help='The number of beams for beam search.')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--early_stopping', type=eval, default=False, help='Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.')
    parser.add_argument('--device', type=str, default='gpu', help='The device to select for training the model.')

    args = parser.parse_args()
    return args
# yapf: enable


def interaction(args, model, tokenizer):
    history = []
    start_info = "Enter [EXIT] to quit the interaction, [NEXT] to start a new conversation."
    cprint(start_info, "yellow", attrs=["bold"])
    while True:
        user_utt = input(colored("[Human]: ", "red", attrs=["bold"])).strip()
        if user_utt == "[EXIT]":
            break
        elif user_utt == "[NEXT]":
            history = []
            cprint(start_info, "yellow", attrs=["bold"])
        else:
            history.append(user_utt)
            inputs = tokenizer.dialogue_encode(history,
                                               add_start_token_as_response=True,
                                               return_tensors=True,
                                               is_split_into_words=False)
            inputs['input_ids'] = inputs['input_ids'].astype('int64')
            ids, scores = model.generate(
                input_ids=inputs['input_ids'],
                token_type_ids=inputs['token_type_ids'],
                position_ids=inputs['position_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=args.max_dec_len,
                min_length=args.min_dec_len,
                decode_strategy=args.decode_strategy,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping,
                num_return_sequences=args.num_return_sequences,
                use_faster=True)
            bot_response = select_response(ids,
                                           scores,
                                           tokenizer,
                                           args.max_dec_len,
                                           args.num_return_sequences,
                                           keep_space=False)[0]
            print(colored("[Bot]:", "blue", attrs=["bold"]),
                  colored(bot_response, attrs=["bold"]))
            history.append(bot_response)
    return


def main(args):
    paddle.set_device(args.device)
    if args.seed is not None:
        set_seed(args.seed)

    # Initialize the model and tokenizer
    model_name_or_path = 'plato-mini'
    model = UnifiedTransformerLMHeadModel.from_pretrained(
        args.model_name_or_path)
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(
        args.model_name_or_path)

    model.eval()
    interaction(args, model, tokenizer)


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    main(args)
