import argparse


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train.")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, help="Path to pre-trained bert model or shortcut name.")
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the checkpoints will be saved.")
    parser.add_argument("--data_dir", default=None, type=str, help="The directory where the dataset will be load.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of checkpoint to be loaded.")
    parser.add_argument("--max_seq_len", default=None, type=int, help="The maximum total input sequence length after tokenization for trainng. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--test_max_seq_len", default=None, type=int, help="The maximum total input sequence length after tokenization for testing. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--test_batch_size", default=None, type=int, help="Batch size per GPU/CPU for testing.")
    parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--epochs", default=None, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--logging_steps", default=None, type=int, help="Log every X updates steps.")
    parser.add_argument("--save_steps", default=None, type=int, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="The proportion of warmup.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The max value of grad norm.")
    parser.add_argument("--do_train", default=True, type=eval, help="Whether training.")
    parser.add_argument("--do_eval", default=True, type=eval, help="Whether evaluation.")
    parser.add_argument("--do_test", default=True, type=eval, help="Whether testing.")
    parser.add_argument("--device", type=str, default="gpu", help="Device for selecting for the training.")

    args = parser.parse_args()
    return args
# yapf: enable


def set_default_args(args):
    args.task_name = args.task_name.lower()
    if args.task_name == 'udc':
        if not args.save_steps:
            args.save_steps = 1000
        if not args.logging_steps:
            args.logging_steps = 100
        if not args.epochs:
            args.epochs = 2
        if not args.max_seq_len:
            args.max_seq_len = 210
        if not args.test_batch_size:
            args.test_batch_size = 100
    elif args.task_name == 'dstc2':
        if not args.save_steps:
            args.save_steps = 400
        if not args.logging_steps:
            args.logging_steps = 20
        if not args.epochs:
            args.epochs = 40
        if not args.learning_rate:
            args.learning_rate = 5e-5
        if not args.max_seq_len:
            args.max_seq_len = 256
        if not args.test_max_seq_len:
            args.test_max_seq_len = 512
    elif args.task_name == 'atis_slot':
        if not args.save_steps:
            args.save_steps = 100
        if not args.logging_steps:
            args.logging_steps = 10
        if not args.epochs:
            args.epochs = 50
    elif args.task_name == 'atis_intent':
        if not args.save_steps:
            args.save_steps = 100
        if not args.logging_steps:
            args.logging_steps = 10
        if not args.epochs:
            args.epochs = 20
    elif args.task_name == 'mrda':
        if not args.save_steps:
            args.save_steps = 500
        if not args.logging_steps:
            args.logging_steps = 200
        if not args.epochs:
            args.epochs = 7
    elif args.task_name == 'swda':
        if not args.save_steps:
            args.save_steps = 500
        if not args.logging_steps:
            args.logging_steps = 200
        if not args.epochs:
            args.epochs = 3
    else:
        raise ValueError('Not support task: %s.' % args.task_name)

    if not args.data_dir:
        args.data_dir = './DGU_datasets/' + args.task_name
    if not args.output_dir:
        args.output_dir = './checkpoints/' + args.task_name
    if not args.learning_rate:
        args.learning_rate = 2e-5
    if not args.batch_size:
        args.batch_size = 32
    if not args.test_batch_size:
        args.test_batch_size = args.batch_size
    if not args.max_seq_len:
        args.max_seq_len = 128
    if not args.test_max_seq_len:
        args.test_max_seq_len = args.max_seq_len
