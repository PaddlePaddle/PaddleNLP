import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file",
                        type=str,
                        required=False,
                        default=None,
                        help="Train data path.")
    parser.add_argument("--predict_file",
                        type=str,
                        required=False,
                        default=None,
                        help="Predict data path.")
    parser.add_argument("--model_type",
                        default="convbert",
                        type=str,
                        help="Type of pre-trained model.")
    parser.add_argument(
        "--model_name_or_path",
        default="convbert-base",
        type=str,
        help="Path to pre-trained model or shortcut name of model.")
    parser.add_argument(
        "--output_dir",
        default="outputs",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written. "
        "Default as `outputs`")
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help=
        "Proportion of training steps to perform linear learning rate warmup for."
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help=
        "The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=
        "If null_score - best_non_null is greater than the threshold predict null."
    )
    parser.add_argument("--max_query_length",
                        type=int,
                        default=64,
                        help="Max query length.")
    parser.add_argument("--max_answer_length",
                        type=int,
                        default=30,
                        help="Max answer length.")
    parser.add_argument(
        "--do_lower_case",
        action='store_false',
        help=
        "Whether to lower case the input text. Should be True for uncased models and False for cased models."
    )
    parser.add_argument("--verbose",
                        action='store_true',
                        help="Whether to output verbose log.")
    parser.add_argument(
        "--version_2_with_negative",
        action='store_true',
        help=
        "If true, the SQuAD examples contain some that do not have an answer. If using squad v2.0, it should be set true."
    )
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to train the model.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to predict.")
    parser.add_argument(
        "--scheduler_type",
        default="linear",
        type=str,
        help="scheduler_type.",
    )
    parser.add_argument("--layer_lr_decay",
                        default=1.0,
                        type=float,
                        help="layer_lr_decay")
    args = parser.parse_args()
    return args
