import time
import argparse

import paddle
from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer
from paddlenlp.metrics import BLEU, Distinct
from datasets import load_dataset

from utils import print_args, set_seed, create_data_loader, select_response


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_name_or_path', type=str, default='unified_transformer-12L-cn-luge', help='The path or shortcut name of the pre-trained model.')
    parser.add_argument('--output_path', type=str, default='./predict.txt', help='The file path where the infer result will be saved.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for initialization.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='The maximum sequence length of training.')
    parser.add_argument('--max_response_len', type=int, default=128, help='The maximum response sequence length of training.')
    parser.add_argument('--max_knowledge_len', type=int, default=256, help='The maximum knowledge sequence length of training.')
    parser.add_argument('--min_dec_len', type=int, default=1, help='The minimum sequence length of generation.')
    parser.add_argument('--max_dec_len', type=int, default=64, help='The maximum sequence length of generation.')
    parser.add_argument('--num_return_sequences', type=int, default=20, help='The numbers of returned sequences for one input in generation.')
    parser.add_argument('--decode_strategy', type=str, default='sampling', help='The decode strategy in generation.')
    parser.add_argument('--top_k', type=int, default=0, help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p', type=float, default=1.0, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', type=int, default=0, help='The number of beams for beam search.')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--early_stopping', type=eval, default=False, help='Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.')
    parser.add_argument('--device', type=str, default='gpu', help='The device to select for training the model.')
    parser.add_argument('--faster', action='store_true', help='Whether to process inference using faster transformer. ')
    parser.add_argument('--use_fp16_decoding', action='store_true', help='Whether to use fp16 when using faster transformer. Only works when using faster transformer. ')

    args = parser.parse_args()
    return args
# yapf: enable


def calc_bleu_and_distinct(preds, targets):
    assert len(preds) == len(targets), (
        'The length of pred_responses should be equal to the length of '
        'target_responses. But received {} and {}.'.format(
            len(preds), len(targets)))
    bleu1 = BLEU(n_size=1)
    bleu2 = BLEU(n_size=2)
    distinct1 = Distinct(n_size=1)
    distinct2 = Distinct(n_size=2)
    for pred, target in zip(preds, targets):
        pred_tokens = pred.split()
        target_token = target.split()

        bleu1.add_inst(pred_tokens, [target_token])
        bleu2.add_inst(pred_tokens, [target_token])

        distinct1.add_inst(pred_tokens)
        distinct2.add_inst(pred_tokens)

    print('\n' + '*' * 15)
    print('The auto evaluation result is:')
    print('BLEU-1:', bleu1.score())
    print('BLEU-2:', bleu2.score())
    print('DISTINCT-1:', distinct1.score())
    print('DISTINCT-2:', distinct2.score())


@paddle.no_grad()
def infer(args):
    paddle.set_device(args.device)
    set_seed(args.seed)

    model = UnifiedTransformerLMHeadModel.from_pretrained(
        args.model_name_or_path)
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(
        args.model_name_or_path)

    test_ds = load_dataset('duconv', split='test_1')
    test_ds, test_data_loader = create_data_loader(test_ds, tokenizer, args,
                                                   'test')

    model.eval()
    total_time = 0.0
    start_time = time.time()
    pred_responses = []
    for step, inputs in enumerate(test_data_loader, 1):
        input_ids, token_type_ids, position_ids, attention_mask, seq_len = inputs
        output = model.generate(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                seq_len=seq_len,
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
                                use_fp16_decoding=args.use_fp16_decoding,
                                use_faster=args.faster)

        total_time += (time.time() - start_time)
        if step % args.logging_steps == 0:
            print('step %d - %.3fs/step' %
                  (step, total_time / args.logging_steps))
            total_time = 0.0

        ids, scores = output
        results = select_response(ids, scores, tokenizer, args.max_dec_len,
                                  args.num_return_sequences)
        pred_responses.extend(results)

        start_time = time.time()

    with open(args.output_path, 'w', encoding='utf-8') as fout:
        for response in pred_responses:
            fout.write(response + '\n')
    print('\nSave inference result into: %s' % args.output_path)

    target_responses = [example['response'] for example in test_ds]
    calc_bleu_and_distinct(pred_responses, target_responses)


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    infer(args)
