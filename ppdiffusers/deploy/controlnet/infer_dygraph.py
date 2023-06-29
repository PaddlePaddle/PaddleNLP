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

import cv2
import numpy as np
import paddle
from PIL import Image
from tqdm.auto import trange

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.utils.log import logger
from ppdiffusers import ControlNetModel, DiffusionPipeline
from ppdiffusers.utils import load_image

logger.set_level("WARNING")


def get_canny_image(image, args):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv2.Canny(image, args.low_threshold, args.high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the `ppdiffusers` checkpoint to convert (either a local directory or on the bos).",
    )
    parser.add_argument(
        "--controlnet_pretrained_model_name_or_path",
        type=str,
        default="lllyasviel/sd-controlnet-canny",
        help="Path to the `ppdiffusers` controlnet_pretrained_model_name_or_path  checkpoint to convert (either a local directory or on the bos).",
    )
    parser.add_argument("--inference_steps", type=int, default=50, help="The number of unet inference steps.")
    parser.add_argument("--benchmark_steps", type=int, default=1, help="The number of performance benchmark steps.")
    parser.add_argument(
        "--task_name",
        type=str,
        default="text2img_control",
        choices=[
            "text2img_control",
            "img2img_control",
            "inpaint_legacy_control",
            "hiresfix_control",
            "all",
        ],
        help="The task can be one of [text2img_control, img2img_control, inpaint_legacy_control, hiresfix_control, all]. ",
    )
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
        "--guess_mode",
        type=strtobool,
        default=False,
        help="In this mode, the ControlNet encoder will try best to recognize the content of the input image even if you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.",
    )
    parser.add_argument(
        "--attention_type", type=str, default="raw", choices=["raw", "cutlass", "flash", "all"], help="attention_type."
    )
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler-ancestral",
        choices=[
            "pndm",
            "lms",
            "euler",
            "euler-ancestral",
            "dpm-multi",
            "dpm-single",
            "unipc-multi",
            "ddim",
            "ddpm",
            "deis-multi",
            "heun",
            "kdpm2-ancestral",
            "kdpm2",
        ],
        help="The scheduler type of stable diffusion.",
    )
    parser.add_argument("--height", type=int, default=512, help="Height of input image")
    parser.add_argument("--width", type=int, default=512, help="Width of input image")
    parser.add_argument("--hr_resize_height", type=int, default=768, help="HR Height of input image")
    parser.add_argument("--hr_resize_width", type=int, default=768, help="HR Width of input image")
    parser.add_argument("--low_threshold", type=int, default=100, help="The value of Canny low threshold.")
    parser.add_argument("--high_threshold", type=int, default=200, help="The value of Canny high threshold.")
    return parser.parse_args()


def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")
    seed = 1024
    paddle_dtype = paddle.float16 if args.use_fp16 else paddle.float32
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_pretrained_model_name_or_path, paddle_dtype=paddle_dtype
    )
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        paddle_dtype=paddle_dtype,
        custom_pipeline="stable_diffusion_mega",
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.change_scheduler(args.scheduler)
    parse_prompt_type = args.parse_prompt_type

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
        guess_mode = args.guess_mode
        width = args.width
        height = args.height
        hr_resize_width = args.hr_resize_width
        hr_resize_height = args.hr_resize_height
        folder = f"attn_{attention_type}_fp16" if args.use_fp16 else f"attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)
        if args.task_name in ["text2img_control", "all"]:
            init_image = load_image(
                "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png"
            )
            controlnet_cond = get_canny_image(init_image, args)
            # text2img
            prompt = "bird"
            time_costs = []
            # warmup
            pipe.text2img(
                prompt,
                num_inference_steps=10,
                height=height,
                width=width,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                guess_mode=guess_mode,
                parse_prompt_type=parse_prompt_type,
            )
            print("==> Test text2img_control performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.text2img(
                    prompt,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    guess_mode=guess_mode,
                    parse_prompt_type=parse_prompt_type,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/text2img_control.png")

        if args.task_name in ["img2img_control", "all"]:
            img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
            init_image = load_image(img_url)
            controlnet_cond = get_canny_image(init_image, args)
            prompt = "A fantasy landscape, trending on artstation"
            time_costs = []
            # warmup
            pipe.img2img(
                prompt,
                image=init_image,
                num_inference_steps=20,
                height=height,
                width=width,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                guess_mode=guess_mode,
                parse_prompt_type=parse_prompt_type,
            )
            print("==> Test img2img_control performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.img2img(
                    prompt,
                    image=init_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    guess_mode=guess_mode,
                    parse_prompt_type=parse_prompt_type,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/img2img_control.png")

        if args.task_name in ["inpaint_legacy_control", "all"]:
            img_url = (
                "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
            )
            mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
            init_image = load_image(img_url)
            mask_image = load_image(mask_url)
            controlnet_cond = get_canny_image(init_image, args)
            prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            time_costs = []
            task_name = "inpaint_legacy_control"
            pipe.inpaint_legacy(
                prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=20,
                height=height,
                width=width,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                guess_mode=guess_mode,
                parse_prompt_type=parse_prompt_type,
            )
            print(f"==> Test {task_name} performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.inpaint_legacy(
                    prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    guess_mode=guess_mode,
                    parse_prompt_type=parse_prompt_type,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/{task_name}.png")

        if args.task_name in ["hiresfix_control", "all"]:
            # hiresfix_control
            init_image = load_image(
                "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png"
            )
            controlnet_cond = get_canny_image(init_image, args)
            # hiresfix_control
            prompt = "a red bird"
            time_costs = []
            # warmup
            pipe.hires_fix(
                prompt,
                height=height,
                width=width,
                num_inference_steps=20,
                hires_ratio=0.5,
                hr_resize_width=hr_resize_width,
                hr_resize_height=hr_resize_height,
                enable_hr=True,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                guess_mode=guess_mode,
                parse_prompt_type=parse_prompt_type,
            )
            print("==> Test hiresfix_control performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.hires_fix(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=args.inference_steps,
                    hires_ratio=0.5,
                    hr_resize_width=hr_resize_width,
                    hr_resize_height=hr_resize_height,
                    enable_hr=True,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    guess_mode=guess_mode,
                    parse_prompt_type=parse_prompt_type,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/hiresfix_control.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
