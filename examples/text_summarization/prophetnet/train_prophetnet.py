import argparse
import os

import paddle
from paddle.io import DataLoader
from tqdm import tqdm

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers.prophetnet.modeling import ProphetNetForConditionalGeneration
from paddlenlp.transformers.prophetnet.tokenizer import ProphetNetTokenizer

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--dataset",
                    default="gigaword",
                    choices=["cnndm", "gigaword"],
                    type=str,
                    help="Path to tokenizer vocab file. ")
parser.add_argument("--model_name_or_path",
                    default="prophetnet-large-uncased",
                    type=str,
                    required=True,
                    help="Path to pre-trained model. ")
parser.add_argument("--batch_size", default=24, type=int)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--warmup_init_lr", default=1e-07, type=float)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--clip_norm", default=0.1, type=float)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--output_dir", default="./ckpt/gigaword", type=str)

args = parser.parse_args()


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


train_data_src = 'data/' + args.dataset + '_data/uncased_tok_data/train.src'
train_data_tgt = 'data/' + args.dataset + '_data/uncased_tok_data/train.tgt'

dev_data_src = 'data/' + args.dataset + '_data/uncased_tok_data/dev.src'
dev_data_tgt = 'data/' + args.dataset + '_data/uncased_tok_data/dev.tgt'

train_dataset = load_dataset(read,
                             data_path=[train_data_src, train_data_tgt],
                             lazy=False)

dev_dataset = load_dataset(read,
                           data_path=[dev_data_src, dev_data_tgt],
                           lazy=False)

t = ProphetNetTokenizer.from_pretrained(args.model_name_or_path)


class InverseSquareRootSchedule(paddle.optimizer.lr.LRScheduler):

    def __init__(self,
                 warmup_init_lr,
                 warmup_end_lr,
                 warmup_steps,
                 last_epoch=-1,
                 verbose=False):
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_steps
        self.decay_factor = warmup_end_lr * warmup_steps**0.5
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        super(InverseSquareRootSchedule, self).__init__(warmup_init_lr,
                                                        last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            self.base_lr = self.warmup_init_lr + self.last_epoch * self.lr_step
        else:
            self.base_lr = self.decay_factor * self.last_epoch**-0.5
        return self.base_lr


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
            return src_ids, src_attention_mask_ids

    return warpper


trunc = convert_example()

train_dataset = train_dataset.map(trunc)
dev_dataset = dev_dataset.map(trunc)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=t.pad_token_id),  # src_ids
    Pad(axis=0, pad_val=0),  # src_pids
    Pad(axis=0, pad_val=t.pad_token_id),  # tgt_ids
    Pad(axis=0, pad_val=0),  # tgt_pids
    Pad(axis=0, pad_val=t.pad_token_id)  # label
): fn(samples)

batch_size = args.batch_size

train_data_loader = DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               collate_fn=batchify_fn,
                               use_shared_memory=False,
                               num_workers=args.num_workers)

dev_data_loader = DataLoader(dataset=dev_dataset,
                             batch_size=batch_size * 2,
                             shuffle=True,
                             collate_fn=batchify_fn,
                             use_shared_memory=False,
                             num_workers=args.num_workers)

epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay
warmup_init_lr = args.warmup_init_lr
warmup_steps = args.warmup_steps
clip_norm = args.clip_norm
output_dir = args.output_dir

best_valid_loss = None
start_epoch = 0

model = ProphetNetForConditionalGeneration.from_pretrained(
    args.model_name_or_path)

lr_scheduler = InverseSquareRootSchedule(warmup_init_lr, lr, warmup_steps)

optimizer = paddle.optimizer.Adam(learning_rate=lr_scheduler,
                                  parameters=model.parameters(),
                                  weight_decay=weight_decay,
                                  grad_clip=paddle.nn.ClipGradByNorm(clip_norm))

accumulate_batchs_num = int(32 * 16 / batch_size)

scaler = paddle.amp.GradScaler(init_loss_scaling=1024)


def compute_loss(model, logits, labels, ignore_index=-100):
    expend_targets = paddle.cast(paddle.zeros(
        (model.prophetnet.config["ngram"], labels.shape[0],
         labels.shape[1])).fill_(ignore_index),
                                 dtype=paddle.int32)

    for i in range(model.prophetnet.config["ngram"]):
        if i > 0 and model.prophetnet.disable_ngram_loss:
            break
        expend_targets[i, :, :] = labels.cast(dtype=paddle.int32)  # B,Ngram,Seq

    logits = logits.transpose([1, 0, 2, 3])

    if model.prophetnet.eps > 0.0:
        expend_targets_mask = paddle.cast(expend_targets != ignore_index,
                                          dtype=paddle.float32)
        expend_targets = paddle.nn.functional.one_hot(
            expend_targets, num_classes=model.vocab_size)
        expend_targets = paddle.nn.functional.label_smooth(
            expend_targets, epsilon=model.prophetnet.eps)
        loss = paddle.nn.functional.cross_entropy(logits,
                                                  expend_targets,
                                                  soft_label=True,
                                                  reduction='none').squeeze()
        loss = paddle.sum(
            expend_targets_mask * loss) / expend_targets_mask.sum()
    else:
        loss = paddle.nn.functional.cross_entropy(
            logits,
            expend_targets.cast(dtype=paddle.int64),
            ignore_index=ignore_index)

    return loss


@paddle.no_grad()
def valid(data):
    model.eval()
    losses = 0
    with tqdm(total=len(data)) as bar:
        for step, batch in enumerate(data, start=1):
            src_ids, src_attention_mask_ids, decoder_input_ids, decoder_input_attention_mask_ids, label_ids = batch
            src_ids = src_ids.cast(dtype=paddle.int32)
            src_attention_mask_ids = src_attention_mask_ids.cast(
                dtype=paddle.int32)
            decoder_input_ids = decoder_input_ids.cast(dtype=paddle.int32)
            decoder_input_attention_mask_ids = decoder_input_attention_mask_ids.cast(
                dtype=paddle.int32)
            label_ids = label_ids.cast(dtype=paddle.int64)
            _, _, logits = model(
                input_ids=src_ids,
                attention_mask=src_attention_mask_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_input_attention_mask_ids)
            loss = compute_loss(model,
                                logits,
                                label_ids,
                                ignore_index=model.padding_idx)
            losses += loss.detach().numpy()
            bar.update(1)
    return losses / step


def train():
    global_step = 1
    global best_valid_loss
    model.train()
    for epoch in range(start_epoch, epochs):
        with tqdm(total=int(len(train_data_loader) /
                            accumulate_batchs_num)) as train_bar:
            for step, batch in enumerate(train_data_loader, start=1):
                src_ids, src_attention_mask_ids, decoder_input_ids, decoder_input_attention_mask_ids, label_ids = batch
                src_ids = src_ids.cast(dtype=paddle.int32)
                src_attention_mask_ids = src_attention_mask_ids.cast(
                    dtype=paddle.int32)
                decoder_input_ids = decoder_input_ids.cast(dtype=paddle.int32)
                decoder_input_attention_mask_ids = decoder_input_attention_mask_ids.cast(
                    dtype=paddle.int32)
                label_ids = label_ids.cast(dtype=paddle.int64)
                with paddle.amp.auto_cast():
                    _, _, logits = model(
                        input_ids=src_ids,
                        attention_mask=src_attention_mask_ids,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_input_attention_mask_ids)
                    loss = compute_loss(model,
                                        logits,
                                        label_ids,
                                        ignore_index=model.padding_idx)

                scaled = scaler.scale(loss)
                scaled.backward()
                if (step + 1) % accumulate_batchs_num == 0:
                    scaler.minimize(optimizer, scaled)
                    lr_scheduler.step()
                    optimizer.clear_grad()
                    train_bar.update(1)
                    train_bar.set_description(
                        "global step %d, epoch: %d, batch: %d, loss: %f, lr: %.3e"
                        %
                        (global_step, epoch, step, loss, lr_scheduler.get_lr()))
                global_step += 1

        valid_loss = valid(dev_data_loader)
        best_ckpt_path = os.path.join(output_dir, "model_best.pdparams")
        if best_valid_loss is None:
            best_valid_loss = valid_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        print("valid loss: %f, best valid loss: %f" %
              (valid_loss, best_valid_loss))
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
