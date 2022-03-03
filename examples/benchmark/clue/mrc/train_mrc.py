

from functools import partial
import inspect
import os
import collections
import time
import json
from tqdm import tqdm
import argparse 

from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
import paddle
from paddlenlp.data import Stack, Dict, Pad
from paddlenlp.datasets import load_dataset
import paddlenlp as ppnlp
from data import prepare_train_features, prepare_validation_features
from data import CMRC2018,CrossEntropyLossForSQuAD
from data import read_text

@paddle.no_grad()
def evaluate(model, data_loader):
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
    squad_evaluate(
        examples=data_loader.dataset.data,
        preds=all_predictions,
        is_whitespace_splited=False)
    
    model.train()


def do_train(model, criterion, dev_data_loader,train_data_loader):
    global_step = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            global_step += 1
            input_ids, segment_ids, start_positions, end_positions = batch
            logits = model(input_ids=input_ids, token_type_ids=segment_ids)
            loss = criterion(logits, (start_positions, end_positions))

            if global_step % 100 == 0 :
                print("global step %d, epoch: %d, batch: %d, loss: %.5f" % (global_step, epoch, step, loss))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        evaluate(model=model, data_loader=dev_data_loader) 

    model.save_pretrained('./checkpoint')
    tokenizer.save_pretrained('./checkpoint')


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model_path",
        default="checkpoints",
        type=str,
        help="The  path of the checkpoints .",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    train_ds = load_dataset(read_text,file_name='data/train.json',lazy=False)
    dev_ds= load_dataset(read_text,file_name='data/dev.json',lazy=False)

    max_seq_length = 512
    doc_stride = 128
    MODEL_NAME = "ernie-1.0"
    # 训练过程中的最大学习率
    learning_rate = 3e-5 
    # 训练轮次
    epochs = 3
    # 学习率预热比例
    warmup_proportion = 0.1
    # 权重衰减系数，类似模型正则项策略，避免模型过拟合
    weight_decay = 0.01
    batch_size = 32

    tokenizer=ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
    model=ppnlp.transformers.ErnieForQuestionAnswering.from_pretrained(MODEL_NAME)

    

    train_trans_func = partial(prepare_train_features, 
                            max_seq_length=max_seq_length, 
                            doc_stride=doc_stride,
                            tokenizer=tokenizer)

    train_ds.map(train_trans_func, batched=True)

    dev_trans_func = partial(prepare_validation_features, 
                            max_seq_length=max_seq_length, 
                            doc_stride=doc_stride,
                            tokenizer=tokenizer)
                            
    dev_ds.map(dev_trans_func, batched=True)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=batch_size, shuffle=True)

    train_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "start_positions": Stack(dtype="int64"),
        "end_positions": Stack(dtype="int64")
    }): fn(samples)

    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=train_batchify_fn,
        return_list=True)

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=batch_size, shuffle=False)

    dev_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
    }): fn(samples)


    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=dev_batchify_fn,
        return_list=True)


    num_training_steps = len(train_data_loader) * epochs
    lr_scheduler = ppnlp.transformers.LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
    criterion = CrossEntropyLossForSQuAD()
    # 模型训练
    do_train(model, criterion, dev_data_loader,train_data_loader)