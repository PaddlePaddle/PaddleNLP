# Copyright (c) 2023 torchtorch Authors. All Rights Reserved.
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

import torch

torch.nn.functional.scaled_dot_product_attention_ = torch.nn.functional.scaled_dot_product_attention
delattr(torch.nn.functional, "scaled_dot_product_attention")
import numpy as np
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionImageVariationPipeline,
    UniPCMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from diffusers.utils import load_image
from tqdm.auto import trange


def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def change_scheduler(self, scheduler_type="ddim"):
    self.orginal_scheduler_config = self.scheduler.config
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "pndm":
        scheduler = PNDMScheduler.from_config(self.orginal_scheduler_config, skip_prk_steps=True)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "dpm-multi":
        scheduler = DPMSolverMultistepScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "dpm-single":
        scheduler = DPMSolverSinglestepScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "kdpm2-ancestral":
        scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "kdpm2":
        scheduler = KDPM2DiscreteScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "unipc-multi":
        scheduler = UniPCMultistepScheduler.from_config(self.orginal_scheduler_config)
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_config(
            self.orginal_scheduler_config,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
        )
    elif scheduler_type == "ddpm":
        scheduler = DDPMScheduler.from_config(
            self.orginal_scheduler_config,
        )
    elif scheduler_type == "deis-multi":
        scheduler = DEISMultistepScheduler.from_config(
            self.orginal_scheduler_config,
        )
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
    return scheduler


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument("--inference_steps", type=int, default=50, help="The number of unet inference steps.")
    parser.add_argument("--benchmark_steps", type=int, default=10, help="The number of performance benchmark steps.")
    parser.add_argument(
        "--parse_prompt_type",
        type=str,
        default="raw",
        choices=[
            "raw",
            "lpw",
        ],
        help="The parse_prompt_type can be one of [raw, lpw]. ",
    )
    parser.add_argument("--channels_last", type=strtobool, default=False, help="Wheter to use channels_last")
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument("--tf32", type=strtobool, default=True, help="tf32")
    parser.add_argument("--compile", type=strtobool, default=False, help="compile")
    parser.add_argument(
        "--attention_type",
        type=str,
        default="sdp",
        choices=[
            "raw",
            "sdp",
        ],
        help="attention_type.",
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
    return parser.parse_args()


def attn_processors(self):
    processors = {}

    def fn_recursive_add_processors(name: str, module, processors):
        if hasattr(module, "set_processor"):
            processors[f"{name}.processor"] = module.processor

        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in self.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def set_attn_processor(self, processor):
    count = len(attn_processors(self).keys())

    if isinstance(processor, dict) and len(processor) != count:
        raise ValueError(
            f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
            f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
        )

    def fn_recursive_attn_processor(name: str, module, processor):
        if hasattr(module, "set_processor"):
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in self.named_children():
        fn_recursive_attn_processor(name, module, processor)


def main(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False

    seed = 1024
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=torch_dtype,
    )
    scheduler = change_scheduler(pipe, args.scheduler)
    pipe.scheduler = scheduler
    if args.device_id >= 0:
        pipe.to(f"cuda:{args.device_id}")

    if args.attention_type == "all":
        args.attention_type = ["raw", "sdp"]
    else:
        args.attention_type = [args.attention_type]

    for attention_type in args.attention_type:
        attn_prrocessor_cls = AttnProcessor if attention_type == "raw" else AttnProcessor2_0
        if attention_type == "sdp":
            torch.nn.functional.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention_

        set_attn_processor(pipe.unet, attn_prrocessor_cls())
        set_attn_processor(pipe.vae, attn_prrocessor_cls())
        if args.channels_last:
            pipe.unet.to(memory_format=torch.channels_last)

        if args.compile:
            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        width = args.width
        height = args.height
        pipe.set_progress_bar_config(disable=True)

        folder = f"torch_attn_{attention_type}_fp16" if args.use_fp16 else f"torch_attn_{attention_type}_fp32"
        os.makedirs(folder, exist_ok=True)

        # image_vairation
        img_url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
        )
        init_image = load_image(img_url).resize((width, height))
        time_costs = []
        # warmup
        pipe(
            image=init_image,
            num_inference_steps=20,
            height=height,
            width=width,
        )
        print("==> Test image_vairation performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            torch.cuda.manual_seed(seed)
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
        images[0].save(f"{folder}/image_vairation.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
