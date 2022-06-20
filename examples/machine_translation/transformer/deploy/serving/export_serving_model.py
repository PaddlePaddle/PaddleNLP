import argparse
import paddle
import paddle_serving_client.io as serving_io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="input inference model dir")
    return parser.parse_args()


def do_export(model_dir):
    feed_names, fetch_names = serving_io.inference_model_to_serving(
        dirname=model_dir,
        serving_server="transformer_server",
        serving_client="transformer_client",
        model_filename="transformer.pdmodel",
        params_filename="transformer.pdiparams")

    print("model feed_names : %s" % feed_names)
    print("model fetch_names : %s" % fetch_names)


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    do_export(args.model_dir)
