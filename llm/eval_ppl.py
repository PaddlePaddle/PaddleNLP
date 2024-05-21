import numpy as np
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
)
from tqdm import tqdm
import paddle
import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--device', type=str, default="gpu:7", help='device: gpu or cpu')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/home/song_test/PaddleNLP/llm/trained_model/llama_lora_merge/")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=4096, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--sliding_window', type=int, default=256, help='')
    parser.add_argument('--data_path', type=str, default="/home/LongLoRA/eval_dataset/Proof-pile/test_sampled_data.bin", help='')
    args = parser.parse_args()
    return args

def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = paddle.stack([paddle.to_tensor((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = paddle.stack([paddle.to_tensor((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, blocking=False), y.pin_memory().to(device, blocking=False)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y

def evaluate(model, data, args):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    with paddle.no_grad():
        print(f"Using seq length {args.seq_len}")
        paddle.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'],
                    args.seq_len,
                    args.batch_size,
                    device=args.device,
                    sliding_window=args.sliding_window
                )
            ),
            total = iceildiv(
                iceildiv(len(data['val']), args.sliding_window),
                args.batch_size
            )
        ):
            val_loss = 0.
            acc = 0.
            cnt = 0
            for part_idx, i in enumerate(range(0, x.shape[1], args.seq_len)):
                part_len = x[:, i:i + args.seq_len].shape[1]
                outputs = model(
                    input_ids = x[:, i:i + args.seq_len],
                    labels = x[:, i:i + args.seq_len].contiguous(),
                    use_cache=False
                )

                val_loss = outputs[0] * part_len + val_loss

                acc = ((outputs[1].argmax(-1) == y[:, i:i+args.seq_len]).sum()) + acc
                cnt += part_len

                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs[0].item())
            val_loss /= cnt
            acc /= cnt
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())
    stats['val_acc'] = paddle.to_tensor(acc_list).mean().item()
    stats['val_loss'] = paddle.to_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = paddle.exp(paddle.to_tensor(loss_step_list_val).mean(axis=1))

    return stats





def main(args):
    paddle.device.set_device(args.device)
    data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}
    print(f"Num validation tokens: {len(data['val'])}")
    print("data path", args.data_path)
    print("base model", args.base_model)

    config = AutoConfig.from_pretrained(args.base_model, cache_dir=args.cache_dir)

    # context_size = args.context_size if args.context_size > 0 else args.seq_len
    # orig_ctx_len = getattr(config, 'max_position_embeddings', None)

    # if orig_ctx_len and context_size > orig_ctx_len:
    #     scaling_factor = float(math.ceil(context_size / orig_ctx_len))
    # config.rope_scaling = {"type": "linear", "factor": 2}

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
    )
    model.resize_token_embeddings(32001)
    
    stats = evaluate(model, data, args)
    print(stats)

if __name__ == "__main__":

    args = parse_config()
    main(args)
