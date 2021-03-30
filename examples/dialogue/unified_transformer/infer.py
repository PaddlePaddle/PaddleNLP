import time

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer
from paddlenlp.metrics import BLEU, Distinct

from utils import parse_args, print_args, set_seed, create_data_loader, select_response


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

    test_ds = load_dataset('duconv', splits='test_1')
    test_ds, test_data_loader = create_data_loader(test_ds, tokenizer, args,
                                                   'test')

    model.eval()
    total_time = 0.0
    start_time = time.time()
    pred_responses = []
    for step, inputs in enumerate(test_data_loader, 1):
        input_ids, token_type_ids, position_ids, attention_mask = inputs
        ids, scores = model.generate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=args.max_dec_len,
            min_length=args.min_dec_len,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            num_return_sequences=args.num_samples)

        total_time += (time.time() - start_time)
        if step % args.logging_steps == 0:
            print('step %d - %.3fs/step' %
                  (step, total_time / args.logging_steps))
            total_time = 0.0
        results = select_response(ids, scores, tokenizer, args.max_dec_len,
                                  args.num_samples)
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
