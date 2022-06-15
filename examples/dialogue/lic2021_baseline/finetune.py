import os
import time
import math
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.optimizer.lr import NoamDecay
from paddle.optimizer import AdamW

from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer

from args import parse_args, print_args
from data import DialogueDataset


def save_ckpt(model, tokenizer, save_dir, name):
    output_dir = os.path.join(save_dir, "model_{}".format(name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Need better way to get inner model of DataParallel
    model_to_save = model._layers if isinstance(model,
                                                paddle.DataParallel) else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main(args):
    paddle.set_device(args.device)
    paddle.seed(args.seed)
    world_size = dist.get_world_size()
    if world_size > 1:
        dist.init_parallel_env()

    model = UnifiedTransformerLMHeadModel.from_pretrained(
        args.model_name_or_path)
    tokenizer = UnifiedTransformerTokenizer.from_pretrained(
        args.model_name_or_path)
    if world_size > 1:
        model = paddle.DataParallel(model)

    train_dataset = DialogueDataset(args.train_data_path,
                                    args.batch_size,
                                    tokenizer.pad_token_id,
                                    tokenizer.cls_token_id,
                                    args.sort_pool_size,
                                    args.seed,
                                    mode='train')
    train_dataloader = DataLoader(train_dataset,
                                  return_list=True,
                                  batch_size=None)
    valid_dataset = DialogueDataset(args.valid_data_path,
                                    args.batch_size,
                                    tokenizer.pad_token_id,
                                    tokenizer.cls_token_id,
                                    args.sort_pool_size,
                                    mode='valid')
    valid_dataloader = DataLoader(valid_dataset,
                                  return_list=True,
                                  batch_size=None)

    lr_scheduler = NoamDecay(1 / (args.warmup_steps * (args.lr**2)),
                             args.warmup_steps)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = AdamW(learning_rate=lr_scheduler,
                      parameters=model.parameters(),
                      weight_decay=args.weight_decay,
                      apply_decay_param_fun=lambda x: x in decay_params,
                      grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm))

    step = 0
    total_time = 0.0
    for epoch in range(args.epochs):
        print('\nEpoch %d/%d' % (epoch + 1, args.epochs))
        batch_start_time = time.time()
        for inputs in train_dataloader:
            step += 1
            token_ids, type_ids, pos_ids, generation_mask, tgt_label, tgt_pos = inputs

            logits = model(token_ids, type_ids, pos_ids, generation_mask,
                           tgt_pos)
            loss = F.cross_entropy(logits, tgt_label)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            total_time += (time.time() - batch_start_time)
            if step % args.logging_steps == 0:
                ppl = paddle.exp(loss)
                print(
                    'step %d - loss: %.4f - ppl: %.4f - lr: %.7f - %.3fs/step' %
                    (step, loss, ppl, optimizer.get_lr(),
                     total_time / args.logging_steps))
                total_time = 0.0
            if step % args.save_steps == 0:
                evaluation(model, valid_dataloader)
                if dist.get_rank() == 0:
                    save_ckpt(model, tokenizer, args.save_dir, step)
            batch_start_time = time.time()


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
        token_ids, type_ids, pos_ids, generation_mask, tgt_label, tgt_pos = inputs

        logits = model(token_ids, type_ids, pos_ids, generation_mask, tgt_pos)
        loss = F.cross_entropy(logits, tgt_label, reduction='sum')

        total_loss += loss.numpy()[0]
        total_tokens += tgt_label.shape[0]

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    avg_speed = (time.time() - start_time) / step
    print('loss: %.4f - ppl: %.4f - %.3fs/step\n' % (avg_loss, ppl, avg_speed))
    model.train()


if __name__ == '__main__':
    args = parse_args()
    print_args(args)

    main(args)
