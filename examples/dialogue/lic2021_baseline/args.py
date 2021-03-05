import argparse


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_name_or_path', type=str, default='unified_transformer-12L-cn', help='The path or shortcut name of the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='The directory where the checkpoints will be saved.')
    parser.add_argument('--output_path', type=str, default='./predict.txt', help='The file path where the infer result will be saved.')
    parser.add_argument('--train_data_path', type=str, default='./datasets/train.txt', help='Specify the path to load train data.')
    parser.add_argument('--valid_data_path', type=str, default='./datasets/valid.txt', help='Specify the path to load valid data.')
    parser.add_argument('--test_data_path', type=str, default='./datasets/test.txt', help='Specify the path to load test data.')
    parser.add_argument('--logging_steps', type=int, default=500, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=8000, help='Save checkpoint every X updates steps.')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for initialization.')
    parser.add_argument('--n_gpus', type=int, default=1, help='The number of gpus to use, 0 for cpu.')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--infer_batch_size', type=int, default=4, help='Batch size per GPU/CPU for infer.')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='The weight decay for optimizer.')
    parser.add_argument('--epochs', type=int, default=10, help='Total number of training epochs to perform.')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='The number of warmup steps.')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='The max value of grad norm.')
    parser.add_argument('--sort_pool_size', type=int, default=65536, help='The pool size for sort in build batch data.')
    parser.add_argument('--topk', type=int, default=5, help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--min_dec_len', type=int, default=1, help='The minimum sequence length of generation.')
    parser.add_argument('--max_dec_len', type=int, default=64, help='The maximum sequence length of generation.')
    parser.add_argument('--num_samples', type=int, default=20, help='The decode numbers in generation.')
    parser.add_argument('--decode_strategy', type=str, default='sampling', help='The decode strategy in generation.')

    args = parser.parse_args()
    return args
# yapf: enable


def print_args(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')