import os
import time
import inspect
from functools import partial
import argparse
import collections
import json

import numpy as np
import random
import paddle
from paddle.io import TensorDataset, DataLoader

from paddlenlp.transformers import ErnieForMultipleChoice, ErnieTokenizer
from paddlenlp.transformers import RobertaForMultipleChoice, RobertaTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.transformers import LinearDecayWithWarmup
from CHID_preprocess import RawResult, get_final_predictions, write_predictions, generate_input, evaluate

MODEL_CLASSES = {
    "ernie": (ErnieForMultipleChoice, ErnieTokenizer),
    "roberta": (RobertaForMultipleChoice, RobertaTokenizer)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )

    parser.add_argument(
        "--output_dir",
        default="best_clue_model",
        type=str,
        help="The  path of the checkpoints .", )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="The max value of grad norm.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--logging_steps",
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


def process_train_data(input_dir, tokenizer, max_seq_length, max_num_choices):

    train_file = '../data/chid/train.json'
    train_ans_file = '../data/chid/train_answer.json'

    train_example_file = os.path.join(
        input_dir, 'train_examples_{}.pkl'.format(str(max_seq_length)))
    train_feature_file = os.path.join(
        input_dir, 'train_features_{}.pkl'.format(str(max_seq_length)))

    train_features = generate_input(
        train_file,
        train_ans_file,
        train_example_file,
        train_feature_file,
        tokenizer,
        max_seq_length=max_seq_length,
        max_num_choices=max_num_choices,
        is_training=True)

    print("loaded train dataset")
    print("Num generate examples = {}".format(len(train_features)))

    all_input_ids = paddle.to_tensor(
        [f.input_ids for f in train_features], dtype='int64')
    all_input_masks = paddle.to_tensor(
        [f.input_masks for f in train_features], dtype='int64')
    all_segment_ids = paddle.to_tensor(
        [f.segment_ids for f in train_features], dtype='int64')
    all_choice_masks = paddle.to_tensor(
        [f.choice_masks for f in train_features], dtype='int64')
    all_labels = paddle.to_tensor(
        [f.label for f in train_features], dtype='int64')

    train_data = TensorDataset([
        all_input_ids, all_input_masks, all_segment_ids, all_choice_masks,
        all_labels
    ])

    return train_data


def process_validation_data(input_dir, tokenizer, max_seq_length,
                            max_num_choices):
    predict_file = '../data/chid/dev.json'
    dev_example_file = os.path.join(
        input_dir, 'dev_examples_{}.pkl'.format(str(max_seq_length)))
    dev_feature_file = os.path.join(
        input_dir, 'dev_features_{}.pkl'.format(str(max_seq_length)))

    eval_features = generate_input(
        predict_file,
        None,
        dev_example_file,
        dev_feature_file,
        tokenizer,
        max_seq_length=max_seq_length,
        max_num_choices=max_num_choices,
        is_training=False)

    all_example_ids = [f.example_id for f in eval_features]
    all_tags = [f.tag for f in eval_features]
    all_input_ids = paddle.to_tensor(
        [f.input_ids for f in eval_features], dtype="int64")
    all_input_masks = paddle.to_tensor(
        [f.input_masks for f in eval_features], dtype="int64")
    all_segment_ids = paddle.to_tensor(
        [f.segment_ids for f in eval_features], dtype="int64")
    all_choice_masks = paddle.to_tensor(
        [f.choice_masks for f in eval_features], dtype="int64")
    all_example_index = paddle.arange(all_input_ids.shape[0], dtype="int64")

    eval_data = TensorDataset([
        all_input_ids, all_input_masks, all_segment_ids, all_choice_masks,
        all_example_index
    ])

    return eval_data, all_example_ids, all_tags, eval_features


@paddle.no_grad()
def do_evaluate(model, dev_data_loader, all_example_ids, all_tags,
                eval_features):
    all_results = []
    model.eval()
    output_dir = '../data'
    for step, batch in enumerate(dev_data_loader):
        input_ids, input_masks, segment_ids, choice_masks, example_indices = batch
        batch_logits = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_masks)

        for i, example_index in enumerate(example_indices):
            logits = batch_logits[i].numpy().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    example_id=all_example_ids[unique_id],
                    tag=all_tags[unique_id],
                    logit=logits))

    predict_file = 'dev_predictions.json'
    predict_ans_file = '../data/chid/dev_answer.json'
    print('decoder raw results')
    tmp_predict_file = os.path.join(output_dir, "raw_predictions.pkl")
    output_prediction_file = os.path.join(output_dir, predict_file)
    results = get_final_predictions(all_results, tmp_predict_file, g=True)
    write_predictions(results, output_prediction_file)
    print('predictions saved to {}'.format(output_prediction_file))

    acc = evaluate(predict_ans_file, output_prediction_file)
    print('{predict_file} eval acc：{acc}')
    model.train()
    return acc


def do_train(args, model, train_data_loader, dev_data_loader, all_example_ids,
             all_tags, eval_features):
    model.train()
    global_step = 0
    tic_train = time.time()
    best_acc = 0.0
    tic_reader = time.time()
    reader_time = 0.0
    for epoch in range(args.num_train_epochs):
        metric.reset()
        for step, batch in enumerate(train_data_loader):
            reader_time += time.time() - tic_reader
            input_ids, input_masks, segment_ids, choice_masks, labels = batch
            logits = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            global_step += 1
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, reader: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       args.logging_steps / (time.time() - tic_train),
                       args.logging_steps / reader_time))
                tic_train = time.time()
                acc = do_evaluate(model, dev_data_loader, all_example_ids,
                                  all_tags, eval_features)
                if paddle.distributed.get_rank() == 0 and acc > best_acc:
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    model_to_save.save_pretrained(args.output_dir)
                    best_acc = acc
            tic_reader = time.time()


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    set_seed(args)
    paddle.set_device(args.device)

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    os.makedirs(args.output_dir, exist_ok=True)
    max_num_choices = 10
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    train_data = process_train_data(args.output_dir, tokenizer,
                                    args.max_seq_length, max_num_choices)
    train_data_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, num_workers=0)

    eval_data, all_example_ids, all_tags, eval_features = process_validation_data(
        args.output_dir, tokenizer, args.max_seq_length, max_num_choices)

    # Run prediction for full data
    dev_data_loader = DataLoader(eval_data, batch_size=args.batch_size)

    model = model_class.from_pretrained(
        args.model_name_or_path, num_choices=max_num_choices)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         0)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=grad_clip)

    criterion = paddle.nn.loss.CrossEntropyLoss()

    metric = paddle.metric.Accuracy()

    do_train(args, model, train_data_loader, dev_data_loader, all_example_ids,
             all_tags, eval_features)
