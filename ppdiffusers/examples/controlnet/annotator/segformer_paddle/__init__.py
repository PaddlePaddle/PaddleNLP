# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import cv2
import numpy as np
import paddle
from annotator.util import annotator_ckpts_path
from paddleseg.cvlibs import Config, manager
from paddleseg.transforms import Compose
from paddleseg.utils import get_image_list, get_sys_env, logger
from pydantic import NoneBytes

from .predict import predict, quick_predict


def parse_args():
    parser = argparse.ArgumentParser(description="Model prediction")

    # params of prediction
    parser.add_argument("--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        "--model_path", dest="model_path", help="The path of model for prediction", type=str, default=None
    )
    parser.add_argument(
        "--image_path",
        dest="image_path",
        help="The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="The directory for saving the predicted results",
        type=str,
        default="./output/result",
    )

    # augment for prediction
    parser.add_argument(
        "--aug_pred",
        dest="aug_pred",
        help="Whether to use mulit-scales and flip augment for prediction",
        action="store_true",
    )
    parser.add_argument("--scales", dest="scales", nargs="+", help="Scales for augment", type=float, default=1.0)
    parser.add_argument(
        "--flip_horizontal",
        dest="flip_horizontal",
        help="Whether to use flip horizontally augment",
        action="store_true",
    )
    parser.add_argument(
        "--flip_vertical", dest="flip_vertical", help="Whether to use flip vertically augment", action="store_true"
    )

    # sliding window prediction
    parser.add_argument(
        "--is_slide", dest="is_slide", help="Whether to prediction by sliding window", action="store_true"
    )
    parser.add_argument(
        "--crop_size",
        dest="crop_size",
        nargs=2,
        help="The crop size of sliding window, the first is width and the second is height.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--stride",
        dest="stride",
        nargs=2,
        help="The stride of sliding window, the first is width and the second is height.",
        type=int,
        default=None,
    )

    # custom color map
    parser.add_argument(
        "--custom_color",
        dest="custom_color",
        nargs="+",
        help="Save images with a custom color map. Default: None, use paddleseg's default color map.",
        type=int,
        default=None,
    )

    # set device
    parser.add_argument(
        "--device",
        dest="device",
        help="Device place to be set, which can be GPU, XPU, NPU, CPU",
        default="gpu",
        type=str,
    )

    return parser.parse_args()


custom_color = [
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
]


def get_test_config(cfg, args):

    test_config = cfg.test_config
    if "aug_eval" in test_config:
        test_config.pop("aug_eval")
    if args.aug_pred:
        test_config["aug_pred"] = args.aug_pred
        test_config["scales"] = args.scales

    if args.flip_horizontal:
        test_config["flip_horizontal"] = args.flip_horizontal

    if args.flip_vertical:
        test_config["flip_vertical"] = args.flip_vertical

    if args.is_slide:
        test_config["is_slide"] = args.is_slide
        test_config["crop_size"] = args.crop_size
        test_config["stride"] = args.stride

    if args.custom_color:
        test_config["custom_color"] = args.custom_color

    return test_config


def main(args):
    env_info = get_sys_env()

    if args.device == "gpu" and env_info["Paddle compiled with cuda"] and env_info["GPUs used"]:
        place = "gpu"
    elif args.device == "xpu" and paddle.is_compiled_with_xpu():
        place = "xpu"
    elif args.device == "npu" and paddle.is_compiled_with_npu():
        place = "npu"
    else:
        place = "cpu"

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError("No configuration file specified.")

    cfg = Config(args.cfg)
    cfg.check_sync_info()

    msg = "\n---------------Config Information---------------\n"
    msg += str(cfg)
    msg += "------------------------------------------------"
    logger.info(msg)

    model = cfg.model
    transforms = Compose(cfg.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info("Number of predict images = {}".format(len(image_list)))

    test_config = get_test_config(cfg, args)

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config,
    )


checkpoint_file = (
    "https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b5_cityscapes_1024x1024_160k/model.pdparams"
)


class SegformerDetector:
    def __init__(self):
        segformer_annotator_ckpts_path = os.path.join(annotator_ckpts_path, "segformer_model")
        modelpath = os.path.join(segformer_annotator_ckpts_path, "model.pdparams")
        if not os.path.exists(modelpath):
            from paddlenlp.utils.downloader import get_path_from_url_with_filelock

            get_path_from_url_with_filelock(checkpoint_file, root_dir=segformer_annotator_ckpts_path)
        self.model_path = modelpath

        cfg = "annotator/segformer_paddle/segformer_b5_cityscapes_1024x1024_160k.yml"
        cfg = Config(cfg)
        cfg.check_sync_info()

        msg = "\n---------------Config Information---------------\n"
        msg += str(cfg)
        msg += "------------------------------------------------"
        logger.info(msg)

        self.model = cfg.model
        self.transforms = Compose(cfg.val_transforms)
        args = parse_args()
        self.test_config = get_test_config(cfg, args)

    def __call__(self, img):
        # img= img.swapaxes(0, 2)
        custom_color_flatten = []
        for color in custom_color:
            custom_color_flatten += color

        res_img, pred_mask = quick_predict(
            self.model,
            model_path=self.model_path,
            transforms=self.transforms,
            image_list=[img],
            image_dir=None,
            save_dir="output",
            skip_save=True,
            custom_color=custom_color_flatten,
            **self.test_config,
        )
        pred_mask = cv2.cvtColor(np.asarray(pred_mask.convert("RGB"))[:, :, ::-1], cv2.COLOR_RGB2BGR)

        return pred_mask


if __name__ == "__main__":
    args = parse_args()
    main(args)
