# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import distutils.util
import math

# import os
import random

# import time
from functools import partial
from pprint import pprint

import numpy as np
import paddle

# from paddlenlp.transformers import LinearDecayWithWarmup
from BRIO_model import BRIO, RankingLoss
from data_utils import BrioDataset, collate_mp_brio

# from datasets import load_dataset
# from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from paddle.io import DataLoader, DistributedBatchSampler
from tqdm import tqdm

# from utils import compute_metrics, convert_example, main_process_first
from utils import compute_metrics

# from paddlenlp.data import DataCollatorForSeq2Seq
# from paddlenlp.transformers import (
#     LinearDecayWithWarmup,
#     PegasusChineseTokenizer,
#     PegasusForConditionalGeneration,
# )
from paddlenlp.transformers import PegasusChineseTokenizer

# from paddlenlp.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
        type=str,
        help="Path to pre-trained model. ",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=False,
        default="data/train.json",
        help="Train data path.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=False,
        default="data/test.json",
        help="Eval data path.",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_source_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--min_target_length",
        default=0,
        type=int,
        help="The minimum total sequence length for target text when generating. ",
    )
    parser.add_argument(
        "--max_target_length",
        default=64,
        type=int,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--epoch",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.",
    )
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override epoch.",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.",
    )
    parser.add_argument(
        "--use_amp",
        default=False,
        type=distutils.util.strtobool,
        help="Enable mixed precision training.",
    )
    parser.add_argument(
        "--scale_loss",
        default=2**15,
        type=float,
        help="The value of scale_loss for fp16.",
    )
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


@paddle.no_grad()
def evaluate(model, data_loader, tokenizer, min_target_length, max_target_length):
    model.eval()
    all_preds = []
    all_labels = []
    model = model._layers if isinstance(model, paddle.DataParallel) else model
    for batch in tqdm(data_loader, total=len(data_loader), desc="Eval step"):
        labels = batch.pop("labels").numpy()
        preds = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            min_length=min_target_length,
            max_length=max_target_length,
            use_cache=True,
        )[0]
        all_preds.extend(
            tokenizer.batch_decode(
                preds.numpy(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    rougel = compute_metrics(all_preds, all_labels)
    model.train()
    return rougel


@paddle.no_grad()
def eval(model, dataloader, tokenizer):
    model.eval()
    cnt = 0
    rouge1, rouge2, rougeLsum = 0, 0, 0
    mle_loss = 0
    mle_fn = paddle.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.scoring_mode()

    # accumulate_step = 4
    # max_lr = 2e-3
    # warmup_steps = 10000
    scale = 1
    # margin = 0.001
    # gold_margin = 0
    # gold_weight = 0
    # mle_weight = 0.1
    # rank_weight = 10
    # report_freq = 100
    # eval_interval = 1000

    # scoring
    for (i, batch) in enumerate(dataloader):
        samples = batch["data"]
        output = model(batch["src_input_ids"], batch["candidate_ids"], True, "log", 2.0, adding=0)
        similarity, gold_similarity = output["score"], output["summary_score"]
        similarity = similarity * scale
        gold_similarity = gold_similarity * scale
        similarity = similarity.cpu().numpy()
        probs = output["probs"][:, :-1]  # truncate last token
        gold = batch["candidate_ids"][:, 0, 1:]  # shift right
        mle_loss += mle_fn(probs, gold)
        # if i % 1000 == 0:
        #     print(f"test similarity: {similarity[0]}")
        max_ids = similarity.argmax(1)
        for j in range(similarity.shape[0]):
            cnt += 1
            sample = samples[j]
            r1, r2, rl = compute_metrics(sample["candidates_untok"][max_ids[j]][0], sample["abstract_untok"])
            # score = rouge_scorer.score("\n".join(sample["abstract_untok"]), "\n".join(sents))
            rouge1 += r1
            rouge2 += r2
            rougeLsum += rl
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    mle_loss = mle_loss / cnt

    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeLsum": rougeLsum,
        "mle_loss": mle_loss,
    }


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    tokenizer = PegasusChineseTokenizer.from_pretrained(args.model_name_or_path)
    # train_set = load_dataset("json", data_files=args.train_file, split="train")
    train_set = BrioDataset(
        "data/diverse/train",
        args.model_name_or_path,
        max_len=512,
        max_num=16,
        total_len=args.max_source_length,
    )
    dev_set = BrioDataset(
        "data/diverse/test",
        args.model_name_or_path,
        is_test=True,
        max_len=512,
        is_sorted=False,
        max_num=16,
        total_len=args.max_source_length,
    )
    # dev_set = load_dataset("json", data_files=args.eval_file, split="train")
    # remove_columns = ["content", "title"]
    # trans_func = partial(
    #     convert_example,
    #     text_column="content",
    #     summary_column="title",
    #     tokenizer=tokenizer,
    #     max_source_length=args.max_source_length,
    #     max_target_length=args.max_target_length,
    # )
    # with main_process_first(desc="train dataset map pre-processing"):
    #     train_set = train_set.map(trans_func, batched=True, load_from_cache_file=True, remove_columns=remove_columns)
    # with main_process_first(desc="dev dataset map pre-processing"):
    #     dev_set = dev_set.map(trans_func, batched=True, load_from_cache_file=True, remove_columns=remove_columns)

    # model = PegasusForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = BRIO(args.model_name_or_path, tokenizer.pad_token_id)
    model.scoring_mode()

    # batchify_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    batchify_fn = partial(collate_mp_brio, pad_token_id=tokenizer.pad_token_id, is_test=False)
    batchify_fn_dev = partial(collate_mp_brio, pad_token_id=tokenizer.pad_token_id, is_test=True)
    # collate_fn_val = partial(collate_mp_brio, pad_token_id=tokenizer.pad_token_id, is_test=True)

    train_batch_sampler = DistributedBatchSampler(train_set, batch_size=args.train_batch_size, shuffle=False)
    dev_batch_sampler = DistributedBatchSampler(dev_set, batch_size=args.eval_batch_size, shuffle=False)
    # dev_gen_batch_sampler = DistributedBatchSampler(dev_set, batch_size=args.eval_batch_size, shuffle=False)
    train_data_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        collate_fn=batchify_fn,
        return_list=True,
    )
    dev_data_loader = DataLoader(
        dataset=dev_set,
        batch_sampler=dev_batch_sampler,
        num_workers=0,
        collate_fn=batchify_fn_dev,
        return_list=True,
    )
    # dev_gen_data_loader = DataLoader(
    #     dataset=dev_set, batch_sampler=dev_gen_batch_sampler, num_workers=0, collate_fn=batchify_fn_dev, return_list=True
    # )

    # dev_batch_sampler = BatchSampler(dev_set, batch_size=args.eval_batch_size, shuffle=False)
    # dev_data_loader = DataLoader(
    #     dataset=dev_set, batch_sampler=dev_batch_sampler, num_workers=0, collate_fn=batchify_fn, return_list=True
    # )
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(num_training_steps / len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.epoch
        num_train_epochs = args.epoch

    # warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    # lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    # decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    # optimizer = paddle.optimizer.AdamW(
    #     learning_rate=lr_scheduler,
    #     beta1=0.9,
    #     beta2=0.999,
    #     epsilon=args.adam_epsilon,
    #     parameters=model.parameters(),
    #     weight_decay=args.weight_decay,
    #     apply_decay_param_fun=lambda x: x in decay_params,
    # )
    optimizer = paddle.optimizer.Adam(parameters=model.parameters())

    mle_fn = paddle.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # if args.use_amp:
    #     scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    def eval_fn(rouge1, rouge2, rougeLsum):
        return 1 - (rouge1 * rouge2 + rougeLsum) / 3

    accumulate_step = 4
    max_lr = 2e-3
    warmup_steps = 10000
    scale = 1
    margin = 0.001
    gold_margin = 0
    gold_weight = 0
    mle_weight = 0.1
    rank_weight = 10
    report_freq = 100
    eval_interval = 5

    # minimum_ranking_loss = 100
    # minimum_mle_loss = 1e5
    global_step = 0
    # best_rougel = 0
    # tic_train = time.time()

    for epoch in range(num_train_epochs):
        avg_ranking_loss = 0
        avg_mle_loss = 0
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        for step, batch in enumerate(train_data_loader):
            # global_step += 1
            step_cnt += 1
            # TODO:amp.auto_cast & use_amp
            output = model(
                batch["src_input_ids"],
                batch["candidate_ids"],
                normalize=True,
                score_mode="log",
                length_penalty=2.0,
                adding=0,
            )
            similarity, gold_similarity = output["score"], output["summary_score"]
            similarity = similarity * scale
            gold_similarity = gold_similarity * scale
            ranking_loss = RankingLoss(similarity, gold_similarity, margin, gold_margin, gold_weight)
            probs = output["probs"][:, :-1]  # truncate last token  [bz, seq_len, word_num]
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            # gold = paddle.repeat_interleave(gold.unsqueeze(2), probs.shape[-1], axis=2)
            mle_loss = mle_fn(probs, gold)
            loss = rank_weight * ranking_loss + mle_weight * mle_loss
            loss = loss / accumulate_step
            avg_loss += loss.item()
            avg_mle_loss += mle_loss.item() / accumulate_step
            avg_ranking_loss += ranking_loss.item() / accumulate_step

            loss.backward()

            if step_cnt == accumulate_step:
                step_cnt = 0
                epoch_step += 1
                global_step += 1
                # adjust learning rate
                lr = max_lr * min(global_step ** (-0.5), global_step * (warmup_steps ** (-1.5)))
                for param_group in optimizer._param_groups:
                    param_group.optimize_attr["learning_rate"] = lr
                optimizer.step()
                optimizer.clear_grad()

            if epoch_step % report_freq == 0 and step_cnt == 0:
                # report state
                print(f"similarity: {similarity[:, :10]}")
                print(f"gold similarity: {gold_similarity}")
                print(
                    "epoch: %d, batch: %d, avg loss: %.6f, avg ranking loss: %.6f, avg mle loss: %.6f"
                    % (
                        epoch + 1,
                        epoch_step,
                        avg_loss / report_freq,
                        avg_ranking_loss / report_freq,
                        avg_mle_loss / report_freq,
                    )
                )
                print(f"learning rate: {lr:.6f}")
                print("loss", {"loss": avg_loss / report_freq}, global_step)
                print("mle_loss", {"loss": avg_mle_loss / report_freq}, global_step)
                print(
                    "ranking_loss",
                    {"loss": avg_ranking_loss / report_freq},
                    global_step,
                )
                avg_mle_loss, avg_ranking_loss, avg_loss = 0, 0, 0
            del similarity, gold_similarity, loss, mle_loss, ranking_loss, output, probs

            if global_step % eval_interval == 0 and global_step != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                # result = eval(dev_data_loader, dev_gen_data_loader, model, args, tokenizer, gpuid, args.do_sample)
                result = eval(model, dev_data_loader, tokenizer)
                loss = eval_fn(result["rouge1"], result["rouge2"], result["rougeLsum"])
                print(loss)
                # if loss < minimum_ranking_loss:
                #     minimum_ranking_loss = loss
                #     if is_mp:
                #         recorder.save(model.module, "model_ranking.bin")
                #     else:
                #         recorder.save(model, "model_ranking.bin")
                #     recorder.print("best ranking loss - epoch: %d, batch: %d" % (epoch, i / args.accumulate_step))
                # if is_master:
                #     recorder.print("val ranking loss: %.6f" % (loss))
                #     recorder.print("val ranking rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                #                    % (result["rouge1"], result["rouge2"], result["rougeLsum"]))
                # # evaluate the model as a generator
                # if args.do_sample:
                #     mle_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
                # else:
                #     mle_loss = result["mle_loss"]
                # if mle_loss < minimum_mle_loss and is_master:
                #     minimum_mle_loss = mle_loss
                #     if is_mp:
                #         recorder.save(model.module, "model_generation.bin")
                #     else:
                #         recorder.save(model, "model_generation.bin")
                #     recorder.print("best generation loss - epoch: %d, batch: %d" % (epoch, i / args.accumulate_step))
                # if is_master:
                #     recorder.print("val generation loss: %.6f" % (mle_loss))
                #     if args.do_sample:
                #         recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                #                        % (result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))
                # # save current model
                # if is_master:
                #     if is_mp:
                #         recorder.save(model.module, "model_cur.bin")
                #     else:
                #         recorder.save(model, "model_cur.bin")
                #     recorder.save(s_optimizer, "optimizer.bin")

            # with paddle.amp.auto_cast(args.use_amp, custom_white_list=["layer_norm", "softmax", "gelu"]):
            #     lm_logits, new_cache, loss = model(**batch)
            # if args.use_amp:
            #     scaled_loss = scaler.scale(loss)
            #     scaled_loss.backward()
            #     scaler.minimize(optimizer, scaled_loss)
            # else:
            #     loss.backward()
            #     optimizer.step()
            # lr_scheduler.step()
            # optimizer.clear_grad()

            # if global_step % args.logging_steps == 0:
            #     logger.info(
            #         "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
            #         % (
            #             global_step,
            #             num_training_steps,
            #             epoch,
            #             step,
            #             paddle.distributed.get_rank(),
            #             loss,
            #             optimizer.get_lr(),
            #             args.logging_steps / (time.time() - tic_train),
            #         )
            #     )
            #     tic_train = time.time()
            # if global_step % args.save_steps == 0 or global_step == num_training_steps:
            #     tic_eval = time.time()
            #     # rougel = evaluate(model, dev_data_loader, tokenizer, args.min_target_length, args.max_target_length)
            #     rougel = evaluate(model, train_data_loader, tokenizer, args.min_target_length, args.max_target_length)
            #     logger.info("eval done total : %s s" % (time.time() - tic_eval))
            #     if paddle.distributed.get_rank() == 0 and best_rougel < rougel:
            #         best_rougel = rougel
            #         output_dir = args.output_dir
            #         if not os.path.exists(output_dir):
            #             os.makedirs(output_dir)
            #         # Need better way to get inner model of DataParallel
            #         model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
            #         model_to_save.save_pretrained(output_dir)
            #         tokenizer.save_pretrained(output_dir)
            # if global_step >= num_training_steps:
            #     return


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    do_train(args)
