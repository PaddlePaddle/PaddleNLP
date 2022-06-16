import sys
import os
import random
import numbers
import logging

import argparse
import paddle
import numpy as np
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForRelationExtraction
from xfun import XFUN

# Todo: delete the following line after the release of v2.2
sys.path.insert(0, "../../../")
logger = logging.getLogger(__name__)


class DataCollator:

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = paddle.to_tensor(data_dict[k])
        return data_dict


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    # yapf: disable
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,)
    parser.add_argument("--train_data_dir", default=None, type=str, required=False,)
    parser.add_argument("--train_label_path", default=None, type=str, required=False,)
    parser.add_argument("--eval_data_dir", default=None, type=str, required=False,)
    parser.add_argument("--eval_label_path", default=None, type=str, required=False,)
    parser.add_argument("--use_vdl", default=False, type=bool, required=False,)
    parser.add_argument("--output_dir", default=None, type=str, required=True,)
    parser.add_argument("--max_seq_length", default=512, type=int,)
    parser.add_argument("--evaluate_during_training", action="store_true",)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for eval.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.",)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.",)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.",)
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.",)
    parser.add_argument("--eval_steps", type=int, default=10, help="eval every X updates steps.",)
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.",)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization",)
    # yapf: enable
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def get_label_maps():
    labels = [
        "O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION",
        "I-HEADER"
    ]
    label2id_map = {label: idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def cal_metric(re_preds, re_labels, entities):
    gt_relations = []
    for b in range(len(re_labels)):
        rel_sent = []
        for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
            rel = {}
            rel["head_id"] = head
            rel["head"] = (entities[b]["start"][rel["head_id"]],
                           entities[b]["end"][rel["head_id"]])
            rel["head_type"] = entities[b]["label"][rel["head_id"]]

            rel["tail_id"] = tail
            rel["tail"] = (entities[b]["start"][rel["tail_id"]],
                           entities[b]["end"][rel["tail_id"]])
            rel["tail_type"] = entities[b]["label"][rel["tail_id"]]

            rel["type"] = 1
            rel_sent.append(rel)
        gt_relations.append(rel_sent)
    re_metrics = re_score(re_preds, gt_relations, mode="boundaries")
    return re_metrics


def re_score(pred_relations, gt_relations, mode="strict"):
    """Evaluate RE predictions

    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations

            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}

        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries'"""

    assert mode in ["strict", "boundaries"]

    relation_types = [v for v in [0, 1] if not v == 0]
    scores = {
        rel: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
        for rel in relation_types + ["ALL"]
    }

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {(rel["head"], rel["head_type"], rel["tail"],
                              rel["tail_type"])
                             for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["head_type"], rel["tail"],
                            rel["tail_type"])
                           for rel in gt_sent if rel["type"] == rel_type}

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                pred_rels = {(rel["head"], rel["tail"])
                             for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"])
                           for rel in gt_sent if rel["type"] == rel_type}

            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    # Compute per entity Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = scores[rel_type]["tp"] / (
                scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = scores[rel_type]["tp"] / (
                scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = (
                2 * scores[rel_type]["p"] * scores[rel_type]["r"] /
                (scores[rel_type]["p"] + scores[rel_type]["r"]))
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean(
        [scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean(
        [scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean(
        [scores[ent_type]["r"] for ent_type in relation_types])

    logger.info(f"RE Evaluation in *** {mode.upper()} *** mode")

    logger.info(
        "processed {} sentences with {} relations; found: {} relations; correct: {}."
        .format(n_sents, n_rels, n_found, tp))
    logger.info("\tALL\t TP: {};\tFP: {};\tFN: {}".format(
        scores["ALL"]["tp"], scores["ALL"]["fp"], scores["ALL"]["fn"]))
    logger.info(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".
        format(precision, recall, f1))
    logger.info(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n"
        .format(scores["ALL"]["Macro_p"], scores["ALL"]["Macro_r"],
                scores["ALL"]["Macro_f1"]))

    for rel_type in relation_types:
        logger.info(
            "\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}"
            .format(
                rel_type,
                scores[rel_type]["tp"],
                scores[rel_type]["fp"],
                scores[rel_type]["fn"],
                scores[rel_type]["p"],
                scores[rel_type]["r"],
                scores[rel_type]["f1"],
                scores[rel_type]["tp"] + scores[rel_type]["fp"],
            ))

    return scores


def evaluate(model, eval_dataloader, logger, prefix=""):
    # Eval!
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataloader.dataset)}")

    re_preds = []
    re_labels = []
    entities = []
    eval_loss = 0.0
    model.eval()
    for idx, batch in enumerate(eval_dataloader):
        with paddle.no_grad():
            outputs = model(**batch)
            loss = outputs['loss'].mean().item()
            if paddle.distributed.get_rank() == 0:
                logger.info(
                    f"[Eval] process: {idx}/{len(eval_dataloader)}, loss: {loss:.5f}"
                )

            eval_loss += loss
        re_preds.extend(outputs['pred_relations'])
        re_labels.extend(batch['relations'])
        entities.extend(outputs['entities'])
    re_metrics = cal_metric(re_preds, re_labels, entities)
    re_metrics = {
        "precision": re_metrics["ALL"]["p"],
        "recall": re_metrics["ALL"]["r"],
        "f1": re_metrics["ALL"]["f1"],
    }
    model.train()
    return re_metrics


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args)

    label2id_map, id2label_map = get_label_maps()
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    # dist mode
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)
    base_model = LayoutXLMModel.from_pretrained(args.model_name_or_path)
    model = LayoutXLMForRelationExtraction(base_model, dropout=None)

    # dist mode
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    train_dataset = XFUN(tokenizer,
                         data_dir=args.train_data_dir,
                         label_path=args.train_label_path,
                         label2id_map=label2id_map,
                         img_size=(224, 224),
                         max_seq_len=args.max_seq_length,
                         pad_token_label_id=pad_token_label_id,
                         contains_re=True,
                         add_special_ids=False,
                         return_attention_mask=True,
                         load_mode='all')

    eval_dataset = XFUN(tokenizer,
                        data_dir=args.eval_data_dir,
                        label_path=args.eval_label_path,
                        label2id_map=label2id_map,
                        img_size=(224, 224),
                        max_seq_len=args.max_seq_length,
                        pad_token_label_id=pad_token_label_id,
                        contains_re=True,
                        add_special_ids=False,
                        return_attention_mask=True,
                        load_mode='all')

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.per_gpu_train_batch_size, shuffle=True)
    args.train_batch_size = args.per_gpu_train_batch_size * max(
        1, paddle.distributed.get_world_size())
    train_dataloader = paddle.io.DataLoader(train_dataset,
                                            batch_sampler=train_sampler,
                                            num_workers=8,
                                            use_shared_memory=True,
                                            collate_fn=DataCollator())

    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=DataCollator())

    t_total = len(train_dataloader) * args.num_train_epochs

    # build linear decay with warmup lr sch
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=args.learning_rate,
        decay_steps=t_total,
        end_lr=0.0,
        power=1.0)
    if args.warmup_steps > 0:
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            lr_scheduler,
            args.warmup_steps,
            start_lr=0,
            end_lr=args.learning_rate,
        )
    grad_clip = paddle.nn.ClipGradByNorm(clip_norm=10)
    optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate,
                                      parameters=model.parameters(),
                                      epsilon=args.adam_epsilon,
                                      grad_clip=grad_clip,
                                      weight_decay=args.weight_decay)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * paddle.distributed.get_world_size()}"
    )
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    train_dataloader_len = len(train_dataloader)
    best_metirc = {'f1': 0}
    model.train()

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            # model outputs are always tuple in ppnlp (see doc)
            loss = outputs['loss']
            loss = loss.mean()

            logger.info(
                f"epoch: [{epoch}/{args.num_train_epochs}], iter: [{step}/{train_dataloader_len}], global_step:{global_step}, train loss: {np.mean(loss.numpy())}, lr: {optimizer.get_lr()}"
            )

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # lr_scheduler.step()  # Update learning rate schedule

            global_step += 1

            if (paddle.distributed.get_rank() == 0 and args.eval_steps > 0
                    and global_step % args.eval_steps == 0):
                # Log metrics
                if paddle.distributed.get_rank(
                ) == 0 and args.evaluate_during_training:
                    results = evaluate(model, eval_dataloader, logger)
                    if results['f1'] > best_metirc['f1']:
                        best_metirc = results
                        output_dir = os.path.join(args.output_dir,
                                                  "checkpoint-best")
                        os.makedirs(output_dir, exist_ok=True)
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        paddle.save(
                            args, os.path.join(output_dir, "training_args.bin"))
                        logger.info(f"Saving model checkpoint to {output_dir}")
                    logger.info(f"eval results: {results}")
                    logger.info(f"best_metirc: {best_metirc}")

            if (paddle.distributed.get_rank() == 0 and args.save_steps > 0
                    and global_step % args.save_steps == 0):
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-latest")
                os.makedirs(output_dir, exist_ok=True)
                if paddle.distributed.get_rank() == 0:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(args,
                                os.path.join(output_dir, "training_args.bin"))
                    logger.info(f"Saving model checkpoint to {output_dir}")
    logger.info(f"best_metirc: {best_metirc}")


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    train(args)
