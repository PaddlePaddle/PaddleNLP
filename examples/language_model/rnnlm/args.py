import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path",
                        type=str,
                        default=None,
                        help="all the data for train,valid,test")
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=650,
                        help='hidden_size')
    parser.add_argument('--num_steps', type=int, default=35, help='num steps')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='max grad norm')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--epoch_start_decay',
                        type=int,
                        default=6,
                        help='epoch_start_decay')
    parser.add_argument('--max_epoch', type=int, default=39, help='max_epoch')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='lr_decay')
    parser.add_argument('--base_lr', type=float, default=1.0, help='base_lr')
    parser.add_argument('--init_scale',
                        type=float,
                        default=0.05,
                        help='init_scale')
    parser.add_argument("--init_from_ckpt",
                        type=str,
                        default=None,
                        help="The path of checkpoint to be loaded.")
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    args = parser.parse_args()
    return args
