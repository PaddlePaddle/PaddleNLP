import argparse
import paddle
from paddlenlp.transformers import AutoModelForPretraining
from paddlenlp.utils.log import logger

paddle.set_device("cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    help="The name of pretrained weights in PaddleNLP.")
parser.add_argument("--path",
                    type=str,
                    help="The path of checkpoint to be loaded.")
parser.add_argument("--output_path",
                    type=str,
                    default=None,
                    help="The path of checkpoint to be loaded.")
args = parser.parse_args()


def init_dygraph_with_static(model, static_params_path):
    from paddlenlp.utils.tools import static_params_to_dygraph
    static_tensor_dict = paddle.static.load_program_state(static_params_path)
    return static_params_to_dygraph(model, static_tensor_dict)


def main(args):
    logger.info("Loading model: %s" % args.model)
    model = AutoModelForPretraining.from_pretrained(args.model)
    logger.info("Loading static params and trans paramters...")
    model_dict = init_dygraph_with_static(model, args.path)
    save_name = args.output_path
    if save_name is None:
        save_name = args.model + "_converted.pdparams"
    if not save_name.endswith(".pdparams"):
        save_name += ".pdparams"
    logger.info("Saving converted params to %s" % save_name)
    paddle.save(model_dict, save_name)
    logger.info("New pdparams saved!")


if __name__ == "__main__":
    main(args)
