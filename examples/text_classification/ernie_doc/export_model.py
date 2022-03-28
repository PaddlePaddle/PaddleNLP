from predict import LongDocClassifier
import paddle
import argparse
from paddlenlp.utils.log import logger
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for predicting (In static mode, it should be the same as in model training process.)")
parser.add_argument("--model_name_or_path", type=str, default="ernie-doc-base-zh", help="Pretraining or finetuned model name or path")
parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--memory_length", type=int, default=128, help="Length of the retained previous heads.")
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Select cpu, gpu devices to train model.")
parser.add_argument("--dataset", default="iflytek", choices=["imdb", "iflytek", "thucnews", "hyp"], type=str, help="The training dataset")
parser.add_argument("--static_path", default=None, type=str, help="The path which your static model is at or where you want to save after converting.")

args = parser.parse_args()

paddle.set_device(args.device)
trainer_num = paddle.distributed.get_world_size()
if trainer_num > 1:
    paddle.distributed.init_parallel_env()
rank = paddle.distributed.get_rank()
if rank == 0:
    if os.path.exists(args.model_name_or_path):
        logger.info("init checkpoint from %s" % args.model_name_or_path)

predictor = LongDocClassifier(model_name_or_path=args.model_name_or_path,
                              rank=rank,
                              trainer_num=trainer_num,
                              batch_size=args.batch_size,
                              max_seq_length=args.max_seq_length,
                              memory_len=args.memory_length,
                              static_mode=True,
                              static_path=args.static_path)