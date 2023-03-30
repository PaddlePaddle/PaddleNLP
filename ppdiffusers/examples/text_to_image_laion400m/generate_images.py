# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import random

import paddle
from tqdm.auto import tqdm

from ppdiffusers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
)


def batchify(data, batch_size=16):
    one_batch = []
    for example in data:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            yield one_batch
            one_batch = []
    if one_batch:
        yield one_batch


def generate_images(
    model_name_or_path,
    batch_size=16,
    file="./data/mscoco.en.1k",
    save_path="output",
    seed=42,
    scheduler_type="ddim",
    eta=0.0,
    num_inference_steps=50,
    guidance_scales=[3, 4, 5, 6, 7, 8],
    height=256,
    width=256,
    device="gpu",
):
    paddle.set_device(device)
    pipe = LDMTextToImagePipeline.from_pretrained(model_name_or_path)
    pipe.set_progress_bar_config(disable=True)
    beta_start = pipe.scheduler.beta_start
    beta_end = pipe.scheduler.beta_end
    if scheduler_type == "pndm":
        scheduler = PNDMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            set_alpha_to_one=False,
            steps_offset=1,
            # Make sure the scheduler compatible with PNDM
            skip_prk_steps=True,
        )
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear")
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear"
        )
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            # Make sure the scheduler compatible with DDIM
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
    pipe.scheduler = scheduler
    # read file
    with open(file, "r") as f:
        all_prompt = [p.strip() for p in f.readlines()]

    for cfg in guidance_scales:
        new_save_path = os.path.join(save_path, f"mscoco.en_g{cfg}")
        os.makedirs(new_save_path, exist_ok=True)
        if seed is not None and seed > 0:
            random.seed(seed)
        i = 0
        for batch_prompt in tqdm(batchify(all_prompt, batch_size=batch_size)):
            sd = random.randint(0, 2**32)
            paddle.seed(sd)
            images = pipe(
                batch_prompt,
                guidance_scale=float(cfg),
                eta=eta,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
            )[0]
            for image in images:
                path = os.path.join(new_save_path, "{:05d}_000.png".format(i))
                image.save(path)
                i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="model_name_or_path.")
    parser.add_argument(
        "--file",
        default="./mscoco.en.1k",
        type=str,
        help="eval file.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="ddim",
        type=str,
        choices=["ddim", "lms", "pndm", "euler-ancest"],
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler-ancest']",
    )
    parser.add_argument("--device", default="gpu", type=str, help="device")
    parser.add_argument("--batch_size", default=16, type=int, help="batch_size")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="num_inference_steps")
    parser.add_argument("--save_path", default="output/1.5b_ldm/12w.pd", type=str, help="Path to the output file.")
    parser.add_argument(
        "--guidance_scales", default=[3, 4, 5, 6, 7, 8], nargs="+", type=str, help="guidance_scales list."
    )
    parser.add_argument("--height", default=256, type=int, help="height.")
    parser.add_argument("--width", default=256, type=int, help="width.")
    args = parser.parse_args()
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")
    generate_images(
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        file=args.file,
        save_path=args.save_path,
        seed=args.seed,
        guidance_scales=args.guidance_scales,
        num_inference_steps=args.num_inference_steps,
        scheduler_type=args.scheduler_type,
        height=args.height,
        width=args.width,
        device=args.device,
    )
