import json
import argparse
from collections import namedtuple
from termcolor import colored, cprint

import paddle

from utils.args import parse_args, str2bool
from utils import gen_inputs
from readers.nsp_reader import NSPReader
from readers.plato_reader import PlatoReader
from model import Plato2InferModel


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("Model")
    group.add_argument("--init_from_ckpt", type=str, default="")
    group.add_argument("--vocab_size", type=int, default=8001)
    group.add_argument("--latent_type_size", type=int, default=20)
    group.add_argument("--num_layers", type=int, default=24)

    group = parser.add_argument_group("Task")
    group.add_argument("--is_cn", type=str2bool, default=False)

    args, _ = parser.parse_known_args()
    NSPReader.add_cmdline_args(parser)

    args = parse_args(parser)
    args.batch_size *= args.latent_type_size

    #print(json.dumps(args, indent=2))
    return args


def load_params(model, init_from_ckpt):
    state_dict = paddle.load(init_from_ckpt)
    model.set_state_dict(state_dict)


def interact(args):
    """Inference main function."""
    plato_reader = PlatoReader(args)
    nsp_reader = NSPReader(args)

    if args.num_layers == 24:
        n_head = 16
        hidden_size = 1024
    elif args.num_layers == 32:
        n_head = 32
        hidden_size = 2048
    else:
        raise ValueError('The pre-trained model only support 24 or 32 layers, '
                         'but received num_layers=%d.' % args.num_layers)

    model = Plato2InferModel(nsp_reader, args.num_layers, n_head, hidden_size)
    load_params(model, args.init_from_ckpt)
    model.eval()

    Example = namedtuple("Example", ["src", "data_id"])
    context = []
    start_info = "Enter [EXIT] to quit the interaction, [NEXT] to start a new conversation."
    cprint(start_info, "yellow", attrs=["bold"])
    while True:
        user_utt = input(colored("[Human]: ", "red", attrs=["bold"])).strip()
        if user_utt == "[EXIT]":
            break
        elif user_utt == "[NEXT]":
            context = []
            cprint(start_info, "yellow", attrs=["bold"])
        else:
            context.append(user_utt)
            example = Example(src=" [SEP] ".join(context), data_id=0)
            record = plato_reader._convert_example_to_record(example,
                                                             is_infer=True)
            data = plato_reader._pad_batch_records([record], is_infer=True)
            inputs = gen_inputs(data, args.latent_type_size)
            inputs['tgt_ids'] = inputs['tgt_ids'].astype('int64')
            pred = model(inputs)[0]
            bot_response = pred["response"]
            print(colored("[Bot]:", "blue", attrs=["bold"]),
                  colored(bot_response, attrs=["bold"]))
            context.append(bot_response)
    return


if __name__ == "__main__":
    args = setup_args()
    interact(args)
