import json
import time
from functools import partial
import argparse

import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Dict, Pad
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.datasets import load_dataset
from data import prepare_validation_features
from data import read_text


@paddle.no_grad()
def do_predict(model, data_loader):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids,
                                                       token_type_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, _, _ = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits), False, 20, 30)

    # Can also write all_nbest_json and scores_diff_json files if needed
    with open('cmrc2018_predict.json', "w", encoding='utf-8') as writer:
        writer.write(
            json.dumps(
                all_predictions, ensure_ascii=False, indent=4) + "\n")

    model.train()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default="checkpoints",
        type=str,
        help="The  path of the checkpoints .", )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args.model_path)
    test_ds = load_dataset(read_text, file_name='data/test.json', lazy=False)

    MODEL_NAME = args.model_path
    max_seq_length = 512
    doc_stride = 128
    batch_size = 32
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
    model = ppnlp.transformers.ErnieForQuestionAnswering.from_pretrained(
        MODEL_NAME)
    dev_trans_func = partial(
        prepare_validation_features,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        tokenizer=tokenizer)
    test_ds.map(dev_trans_func, batched=True)
    test_batch_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=batch_size, shuffle=False)
    dev_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        }): fn(samples)
    test_data_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=dev_batchify_fn,
        return_list=True)
    do_predict(model, test_data_loader)
