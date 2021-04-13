import argparse


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data_path", type=str, default="./1-billion-word/training-tokenized-shuffled/*", help="Specify the path to load train data.")
    parser.add_argument("--dev_data_path", type=str, default="./1-billion-word/heldout-tokenized-shuffled/*", help="Specify the path to load dev data.")
    parser.add_argument("--vocab_file", type=str, default="./1-billion-word/vocab-15w.txt", help="Specify the path to load vocab file.")
    parser.add_argument("--save_dir", type=str, default="./checkpoint/", help="Specify the path to save the checkpoints.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--save_freq", type=int, default=100, help="The frequency, in number of steps, to save checkpoint. (default: %(default)d)")
    parser.add_argument("--log_freq", type=int, default=100, help="The frequency, in number of steps, the training logs are printed. (default: %(default)d)")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="The dropout rate.")
    parser.add_argument("--lr", type=float, default=0.2, help="The initial learning rate.")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help='The max grad norm.')
    parser.add_argument("--max_characters_per_token", type=int, default=50, help="The maximum characters number of token in sequence. (default: %(default)d)")
    parser.add_argument("--unroll_steps", type=int, default=20, help="The sentence length after re-cutting in dataset. (default: %(default)d)")
    parser.add_argument("--char_embed_dim", type=int, default=16, help="The dimension of char_embedding table. (default: %(default)d)")
    parser.add_argument("--projection_dim", type=int, default=512, help="The size of rnn hidden unit. (default: %(default)d)")
    parser.add_argument("--num_layers", type=int, default=2, help="The num of rnn layers. (default: %(default)d)")
    parser.add_argument("--num_highways", type=int, default=2, help="The num of highways in CharEncoder. (default: %(default)d)")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device for selecting for the training.")

    args = parser.parse_args()
    return args
# yapf: enable


def print_args(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')
