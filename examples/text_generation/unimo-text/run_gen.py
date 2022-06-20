import os
import time
import math
import argparse
import json

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import LinearDecayWithWarmup
from paddle.optimizer import AdamW

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import UNIMOLMHeadModel, UNIMOTokenizer, BasicTokenizer
from paddlenlp.metrics import BLEU

from gen_utils import print_args, set_seed, create_data_loader, select_sum


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset_name', type=str, default='dureader_qg', help='The name of the dataset to load.')
    parser.add_argument('--model_name_or_path', type=str, default='unimo-text-1.0', help='The path or shortcut name of the pre-trained model.')
    parser.add_argument("--train_file", type=str, required=False, default=None, help="Train data path.")
    parser.add_argument("--predict_file", type=str, required=False, default=None, help="Predict data path.")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='The directory where the checkpoints will be saved.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every X updates steps.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='The initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='The weight decay for optimizer.')
    parser.add_argument('--epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument('--warmup_propotion', type=float, default=0.02, help='The number of warmup steps.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='The max value of grad norm.')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='epsilon')
    parser.add_argument('--max_seq_len', type=int, default=512, help='The maximum sequence length of training.')
    parser.add_argument('--max_dec_len', type=int, default=20, help='The maximum sequence length of decoding.')
    parser.add_argument('--min_dec_len', type=int, default=3, help='The minimal sequence length of decoding.')
    parser.add_argument('--max_target_len', type=int, default=30, help='The maximum target sequence length of training.')
    parser.add_argument('--max_title_len', type=int, default=30, help='The maximum title sequence length of training.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='The numbers of returned sequences for one input in generation.')
    parser.add_argument('--decode_strategy', type=str, default='beam_search', help='The decode strategy in generation.')
    parser.add_argument('--top_k', type=int, default=0, help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p', type=float, default=1.0, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', type=int, default=6, help='The number of beams for beam search.')
    parser.add_argument('--length_penalty', type=float, default=1.2, help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--device', type=str, default='gpu', help='The device to select for training the model.')
    parser.add_argument('--output_path', type=str, default='./predict.txt', help='The file path where the infer result will be saved.')
    parser.add_argument("--do_train", action='store_true', help="Whether to train the model.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to eval and predict.")

    args = parser.parse_args()
    return args
# yapf: enable


def calc_bleu(preds, targets):
    assert len(preds) == len(targets), (
        'The length of pred_responses should be equal to the length of '
        'target_responses. But received {} and {}.'.format(
            len(preds), len(targets)))
    bleu4 = BLEU(n_size=4)
    tokenizer = BasicTokenizer()

    for pred, target in zip(preds, targets):
        pred_tokens = tokenizer.tokenize(pred)
        target_token = tokenizer.tokenize(target)

        bleu4.add_inst(pred_tokens, [target_token])

    print('\n' + '*' * 15)
    print('The auto evaluation result is:')
    print('BLEU-4:', bleu4.score())


def save_ckpt(model, tokenizer, save_dir, name):
    output_dir = os.path.join(save_dir, "model_{}".format(name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Need better way to get inner model of DataParallel
    model_to_save = model._layers if isinstance(model,
                                                paddle.DataParallel) else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def run(args):
    paddle.set_device(args.device)
    world_size = dist.get_world_size()

    if world_size > 1:
        dist.init_parallel_env()
    set_seed(args.seed)

    model = UNIMOLMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer = UNIMOTokenizer.from_pretrained(args.model_name_or_path)

    if world_size > 1:
        model = paddle.DataParallel(model)

    train_ds = load_dataset(args.dataset_name,
                            splits='train',
                            data_files=args.train_file)
    dev_ds = load_dataset(args.dataset_name,
                          splits='dev',
                          data_files=args.predict_file)

    train_ds, train_data_loader = create_data_loader(train_ds, tokenizer, args,
                                                     'train')
    dev_ds, dev_data_loader = create_data_loader(dev_ds, tokenizer, args,
                                                 'test')

    if args.do_train:
        num_training_steps = args.epochs * len(train_data_loader)

        lr_scheduler = LinearDecayWithWarmup(args.learning_rate,
                                             num_training_steps,
                                             args.warmup_propotion)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.

        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        optimizer = AdamW(learning_rate=lr_scheduler,
                          parameters=model.parameters(),
                          weight_decay=args.weight_decay,
                          beta1=args.beta1,
                          beta2=args.beta2,
                          epsilon=args.epsilon,
                          apply_decay_param_fun=lambda x: x in decay_params,
                          grad_clip=paddle.nn.ClipGradByGlobalNorm(
                              args.max_grad_norm))

        step = 0
        total_time = 0.0
        for epoch in range(args.epochs):
            print('\nEpoch %d/%d' % (epoch + 1, args.epochs))
            batch_start_time = time.time()
            for inputs in train_data_loader:
                step += 1
                labels = inputs[-1]
                logits = model(*inputs[:-1])
                labels = paddle.nn.functional.one_hot(
                    labels, num_classes=logits.shape[-1])
                labels = paddle.nn.functional.label_smooth(labels)
                loss = F.cross_entropy(logits, labels, soft_label=True)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                total_time += (time.time() - batch_start_time)
                if step % args.logging_steps == 0:
                    ppl = paddle.exp(loss)
                    print(
                        'step %d - loss: %.4f - ppl: %.4f - lr: %.7f - %.3fs/step'
                        % (step, loss, ppl, optimizer.get_lr(),
                           total_time / args.logging_steps))
                    total_time = 0.0

                if step % args.save_steps == 0 or step >= num_training_steps:
                    if dist.get_rank() == 0:
                        save_ckpt(model, tokenizer, args.save_dir, step)
                        print('Saved step {} model.\n'.format(step))
                        if args.do_predict:
                            model_eval = model._layers if isinstance(
                                model, paddle.DataParallel) else model
                            evaluation(model_eval, dev_data_loader, args,
                                       tokenizer)

                batch_start_time = time.time()

        print('\nTraining completed.')
    elif args.do_predict:
        model_eval = model._layers if isinstance(model,
                                                 paddle.DataParallel) else model
        evaluation(model_eval, dev_data_loader, args, tokenizer)


@paddle.no_grad()
def evaluation(model, data_loader, args, tokenizer):
    print('\nEval begin...')
    model.eval()
    pred_ref = []
    total_time = 0.0
    start_time = time.time()
    for step, inputs in enumerate(data_loader, 1):
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
            num_return_sequences=args.num_return_sequences,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.mask_token_id)

        total_time += (time.time() - start_time)
        if step % args.logging_steps == 0:
            print('step %d - %.3fs/step' %
                  (step, total_time / args.logging_steps))
            total_time = 0.0

        results = select_sum(ids, scores, tokenizer, args.max_dec_len,
                             args.num_return_sequences)
        pred_ref.extend(results)
        start_time = time.time()

    with open(args.output_path, 'w', encoding='utf-8') as fout:
        for ref in pred_ref:
            fout.write(ref + '\n')

    print('\nSave inference result into: %s' % args.output_path)

    if 'target' in data_loader.dataset[0].keys():
        targets = [example['target'] for example in data_loader.dataset]
        calc_bleu(pred_ref, targets)

    model.train()
    return


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    run(args)
