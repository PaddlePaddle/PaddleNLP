import argparse
import paddle
import paddle_serving_client.io as serving_io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_model_dir",
                        type=str,
                        required=True,
                        help="input inference model dir")
    parser.add_argument("--model_file",
                        type=str,
                        required=True,
                        help="input inference model file name")
    parser.add_argument("--params_file",
                        type=str,
                        required=True,
                        help="input inference parameters file name")
    return parser.parse_args()


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    feed_names, fetch_names = serving_io.inference_model_to_serving(
        dirname=args.inference_model_dir,
        serving_server="serving_server",
        serving_client="serving_client",
        model_filename=args.model_file,
        params_filename=args.params_file)
    print("model feed_names : %s" % feed_names)
    print("model fetch_names : %s" % fetch_names)
