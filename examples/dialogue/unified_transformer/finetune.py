import os
import time
import math

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.optimizer.lr import NoamDecay
from paddle.optimizer import AdamW

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer

from utils import parse_args, print_args, set_seed, create_data_loader


def save_ckpt(model, tokenizer, save_dir, name):
    output_dir = os.path.join(save_dir, "model_{}".format(name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Need better way to get inner model of DataParallel
    model_to_save = model._layers if isinstance(model,
                                                paddle.DataParallel) else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train(args):
    paddle.set_device(args.device)
    world_size = dist.get_world_size()
    if world_size > 1:
        dist.init_parallel_env()

    set_seed(args.seed)

    model = UnifiedTransformerLMHeadModel.from_pretrained(
        args.model_name_or_path)
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(
        args.model_name_or_path)

    if world_size > 1:
        model = paddle.DataParallel(model)

    train_ds, dev_ds = load_dataset('duconv', splits=('train', 'dev'))
    train_ds, train_data_loader = create_data_loader(train_ds, tokenizer, args,
                                                     'train')
    dev_ds, dev_data_loader = create_data_loader(dev_ds, tokenizer, args, 'dev')

    lr_scheduler = NoamDecay(1 / (args.warmup_steps * (args.lr**2)),
                             args.warmup_steps)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm))

    step = 0
    total_time = 0.0
    best_ppl = 1e9
    for epoch in range(args.epochs):
        print('\nEpoch %d/%d' % (epoch + 1, args.epochs))
        batch_start_time = time.time()
        for inputs in train_data_loader:
            step += 1
            labels = inputs[-1]

            logits = model(*inputs[:-1])
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            total_time += (time.time() - batch_start_time)
            if step % args.logging_steps == 0:
                ppl = paddle.exp(loss)
                print('step %d - loss: %.4f - ppl: %.4f - lr: %.7f - %.3fs/step'
                      % (step, loss, ppl, optimizer.get_lr(),
                         total_time / args.logging_steps))
                total_time = 0.0
            if step % args.save_steps == 0:
                ppl = evaluation(model, dev_data_loader)
                if dist.get_rank() == 0:
                    save_ckpt(model, tokenizer, args.save_dir, step)
                    if ppl < best_ppl:
                        best_ppl = ppl
                        save_ckpt(model, tokenizer, args.save_dir, 'best')
                        print('Saved step {} as best model.\n'.format(step))
            batch_start_time = time.time()
    print('\nTraining completed.')


@paddle.no_grad()
def evaluation(model, data_loader):
    print('\nEval begin...')
    model.eval()
    total_tokens = 0
    total_loss = 0.0
    start_time = time.time()
    step = 0
    for inputs in data_loader:
        step += 1
        labels = inputs[-1]

        logits = model(*inputs[:-1])
        loss = F.cross_entropy(logits, labels, reduction='sum')

        total_loss += loss.numpy().item()
        total_tokens += labels.shape[0]

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    avg_speed = (time.time() - start_time) / step
    print('loss: %.4f - ppl: %.4f - %.3fs/step' % (avg_loss, ppl, avg_speed))
    model.train()
    return ppl


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    train(args)
