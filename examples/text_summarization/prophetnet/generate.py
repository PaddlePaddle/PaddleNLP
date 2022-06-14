import argparse
import os
import random
import time
from pprint import pprint

import numpy as np
import paddle
from paddle.io import BatchSampler, DataLoader
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers.prophetnet.modeling import ProphetNetForConditionalGeneration
from paddlenlp.transformers.prophetnet.tokenizer import ProphetNetTokenizer

summarization_name_mapping = {"cnn_dailymail": ("article", "highlights")}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_name_or_path",
                        default="prophetnet-large-uncased",
                        type=str,
                        required=True,
                        help="Path to pre-trained model. ")
    parser.add_argument("--dataset",
                        default="gigaword",
                        choices=["cnndm", "gigaword"],
                        type=str,
                        help="Path to tokenizer vocab file. ")
    parser.add_argument(
        '--output_path',
        type=str,
        default='generate.txt',
        help='The file path where the infer result will be saved.')
    parser.add_argument(
        "--max_source_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--min_target_length",
        default=45,
        type=int,
        help=
        "The minimum total sequence length for target text when generating. ")
    parser.add_argument(
        "--max_target_length",
        default=110,
        type=int,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument('--decode_strategy',
                        default='beam_search',
                        type=str,
                        help='The decode strategy in generation.')
    parser.add_argument(
        '--top_k',
        default=2,
        type=int,
        help=
        'The number of highest probability vocabulary tokens to keep for top-k sampling.'
    )
    parser.add_argument('--top_p',
                        default=1.0,
                        type=float,
                        help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams',
                        default=5,
                        type=int,
                        help='The number of beams for beam search.')
    parser.add_argument(
        '--length_penalty',
        default=1.2,
        type=float,
        help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument(
        '--early_stopping',
        default=False,
        type=eval,
        help=
        'Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.'
    )
    parser.add_argument("--diversity_rate",
                        default=0.0,
                        type=float,
                        help="The diversity of beam search. ")
    parser.add_argument(
        "--num_beam_groups",
        default=1,
        type=int,
        help=
        "Number of groups to divide `num_beams` into in order to use DIVERSE BEAM SEARCH."
    )
    parser.add_argument(
        "--repetition_penalty",
        default=1.0,
        type=float,
        help=
        "Number of groups to divide `num_beams` into in order to use DIVERSE BEAM SEARCH."
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for testing or evaluation.")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        default=True,
        type=bool,
        help="Whether to ignore the tokens corresponding to "
        "padded labels in the loss computation or not.",
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def compute_metrics(preds,
                    labels,
                    tokenizer,
                    ignore_pad_token_for_loss=True,
                    compute_rouge_=True):

    def compute_rouge(predictions,
                      references,
                      rouge_types=None,
                      use_stemmer=True):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeLsum"]

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types,
                                          use_stemmer=use_stemmer)
        aggregator = scoring.BootstrapAggregator()

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            aggregator.add_scores(score)
        result = aggregator.aggregate()
        result = {
            key: round(value.mid.fmeasure * 100, 4)
            for key, value in result.items()
        }
        return result

    def post_process_seq(seq,
                         bos_idx,
                         eos_idx,
                         output_bos=False,
                         output_eos=False):
        """
        Post-process the decoded sequence.
        """
        eos_pos = len(seq) - 1
        for i, idx in enumerate(seq):
            if idx == eos_idx:
                eos_pos = i
                break
        seq = [
            idx for idx in seq[:eos_pos + 1]
            if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
        ]
        return seq

    if ignore_pad_token_for_loss:
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds, decoded_labels = [], []
    for pred, label in zip(preds, labels):
        pred_id = post_process_seq(pred, tokenizer.bos_token_id,
                                   tokenizer.eos_token_id)
        label_id = post_process_seq(label, tokenizer.bos_token_id,
                                    tokenizer.eos_token_id)
        decoded_preds.append(tokenizer.convert_ids_to_string(pred_id))
        decoded_labels.append(tokenizer.convert_ids_to_string(label_id))

    if compute_rouge_:
        rouge_result = compute_rouge(decoded_preds, decoded_labels)
        return rouge_result, decoded_preds
    else:
        return decoded_preds, decoded_labels


def read(data_path):
    data_path_src = data_path[0]
    data_path_tgt = data_path[1]
    with open(data_path_src, 'r', encoding='utf-8') as f_d_s:
        src_lines_length = len(f_d_s.readlines())
    with open(data_path_tgt, 'r', encoding='utf-8') as f_d_t:
        tgt_lines_length = len(f_d_t.readlines())
    assert src_lines_length == tgt_lines_length
    with open(data_path_src, 'r', encoding='utf-8') as f_d_s:
        with open(data_path_tgt, 'r', encoding='utf-8') as f_d_t:
            for row_d_s, row_d_t in tqdm(zip(f_d_s, f_d_t),
                                         total=src_lines_length):
                yield {'article': row_d_s, 'highlights': row_d_t}


def convert_example(is_test=False):

    def warpper(example):
        """convert an example into necessary features"""
        tokens = example['article']
        labels = example['highlights']
        src_ids, src_attention_mask_ids = tokens.split("$1$")
        src_ids = [int(i) for i in src_ids.split(" ")]
        src_attention_mask_ids = [
            int(i) for i in src_attention_mask_ids.split(" ")
        ]

        if not is_test:
            labels, decoder_input_attention_mask_ids = labels.split("$1$")
            labels = [int(i) for i in labels.split(" ")]
            decoder_input_attention_mask_ids = [
                int(i) for i in decoder_input_attention_mask_ids.split(" ")
            ]
            decoder_input_ids = [labels[-1]] + labels[:-1]

            return src_ids, src_attention_mask_ids, decoder_input_ids, decoder_input_attention_mask_ids, labels

        else:
            labels, _ = labels.split("$1$")
            labels = [int(i) for i in labels.split(" ")]
            return src_ids, src_attention_mask_ids, labels

    return warpper


@paddle.no_grad()
def generate(args):
    paddle.set_device(args.device)
    tokenizer = ProphetNetTokenizer.from_pretrained(args.model_name_or_path)
    model = ProphetNetForConditionalGeneration.from_pretrained(
        args.model_name_or_path)

    test_data_src = 'data/' + args.dataset + '_data/uncased_tok_data/test.src'
    test_data_tgt = 'data/' + args.dataset + '_data/uncased_tok_data/test.tgt'

    test_dataset = load_dataset(read,
                                data_path=[test_data_src, test_data_tgt],
                                lazy=False)

    trunc = convert_example(is_test=True)

    test_dataset = test_dataset.map(trunc)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=0),  # attn mask
        Pad(axis=0, pad_val=tokenizer.pad_token_id)  # labels
    ): fn(samples)

    batch_sampler = BatchSampler(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_sampler=batch_sampler,
                                  num_workers=0,
                                  collate_fn=batchify_fn,
                                  return_list=True)

    model.eval()
    total_time = 0.0
    start_time = time.time()
    all_preds = []
    all_labels = []
    for step, batch in tqdm(enumerate(test_data_loader),
                            total=len(test_data_loader)):
        input_ids, attention_mask, labels = batch
        preds, _ = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  max_length=args.max_target_length,
                                  min_length=args.min_target_length,
                                  decode_strategy=args.decode_strategy,
                                  top_k=args.top_k,
                                  top_p=args.top_p,
                                  num_beams=args.num_beams,
                                  length_penalty=args.length_penalty,
                                  early_stopping=args.early_stopping,
                                  diversity_rate=args.diversity_rate,
                                  num_beam_groups=args.num_beam_groups,
                                  repetition_penalty=args.repetition_penalty)
        total_time += (time.time() - start_time)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        if step % args.logging_steps == 0:
            print('step %d - %.3fs/step' %
                  (step, total_time / args.logging_steps))
            total_time = 0.0
        start_time = time.time()
    decoded_preds, _ = compute_metrics(all_preds,
                                       all_labels,
                                       tokenizer,
                                       args.ignore_pad_token_for_loss,
                                       compute_rouge_=False)
    if not os.path.exists(
            os.path.abspath(
                os.path.dirname(args.output_path) + os.path.sep + ".")):
        os.makedirs(
            os.path.abspath(
                os.path.dirname(args.output_path) + os.path.sep + "."))
    with open(args.output_path, 'w', encoding='utf-8') as fout:
        for decoded_pred in decoded_preds:
            fout.write(decoded_pred + '\n')
    print('Save generated result into: %s' % args.output_path)


if __name__ == '__main__':
    args = parse_args()
    pprint(args)
    generate(args)
