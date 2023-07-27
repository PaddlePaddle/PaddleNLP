# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import time
import warnings

import numpy as np
import paddle
from tqdm.auto import trange

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.utils.log import logger
from ppdiffusers import StableDiffusionImageVariationPipeline
from ppdiffusers.utils import load_image

logger.set_level("WARNING")


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="runwayml/stable-diffusion-v1-5",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument("--inference_steps", type=int, default=50, help="The number of unet inference steps.")
    parser.add_argument("--benchmark_steps", type=int, default=1, help="The number of performance benchmark steps.")
    parser.add_argument(
        "--parse_prompt_type",
        type=str,
        default="lpw",
        choices=[
            "raw",
            "lpw",
        ],
        help="The parse_prompt_type can be one of [raw, lpw]. ",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument(
        "--attention_type", type=str, default="raw", choices=["raw", "cutlass", "flash", "all"], help="attention_type."
    )
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument("--height", type=int, default=512, help="Height of input image")
    parser.add_argument("--width", type=int, default=512, help="Width of input image")
    parser.add_argument("--hr_resize_height", type=int, default=768, help="HR Height of input image")
    parser.add_argument("--hr_resize_width", type=int, default=768, help="HR Width of input image")
    return parser.parse_args()


def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")

    seed = 1024
    # paddle_dtype = paddle.float16 if args.use_fp16 else paddle.float32
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        args.model_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    # parse_prompt_type = args.parse_prompt_type
    if args.attention_type == "all":
        args.attention_type = ["raw", "cutlass", "flash"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        if attention_type == "raw":
            pipe.disable_xformers_memory_efficient_attention()
        else:
            try:
                pipe.enable_xformers_memory_efficient_attention(attention_type)
            except Exception as e:
                if attention_type == "flash":
                    warnings.warn(
                        "Attention type flash is not supported on your GPU! We need to use 3060、3070、3080、3090、4060、4070、4080、4090、A30、A100 etc."
                    )
                    continue
                else:
                    raise ValueError(e)

        width = args.width
        height = args.height
        # hr_resize_width = args.hr_resize_width
        # hr_resize_height = args.hr_resize_height
        folder = f"attn_{attention_type}_fp16" if args.use_fp16 else f"attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # image_variation
        img_url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
        )
        init_image = load_image(img_url)
        time_costs = []
        # warmup
        pipe(
            image=init_image,
            num_inference_steps=20,
            height=height,
            width=width,
        )
        print("==> Test image_variation performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            images = pipe(
                image=init_image,
                num_inference_steps=args.inference_steps,
                height=height,
                width=width,
            ).images
            latency = time.time() - start
            time_costs += [latency]
            # print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        images[0].save(f"{folder}/image_variation.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
