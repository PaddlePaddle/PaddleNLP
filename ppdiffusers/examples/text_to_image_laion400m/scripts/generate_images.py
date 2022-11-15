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

import os
import argparse
import random
import paddle
from tqdm.auto import tqdm
from ppdiffusers import PNDMScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler, LDMTextToImagePipeline


def generate_images(model_name_or_path,
                    file='./data/mscoco.en.1k',
                    save_path="output",
                    seed=42,
                    scheduler_type="ddim",
                    eta=0.,
                    guidance_scales=[3, 4, 5, 6, 7, 8],
                    device="gpu"):
    paddle.set_device(device)
    pipe = LDMTextToImagePipeline.from_pretrained(model_name_or_path)
    pipe.set_progress_bar_config(disable=True)
    num_train_timesteps = pipe.scheduler.num_train_timesteps
    beta_start = pipe.scheduler.beta_start
    beta_end = pipe.scheduler.beta_end
    if scheduler_type == "pndm":
        scheduler = PNDMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            skip_prk_steps=True,
        )
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler(beta_start=beta_start,
                                         beta_end=beta_end,
                                         beta_schedule="scaled_linear")
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear")
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
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
        if seed is not None:
            random.seed(seed)
        for i, prompt in tqdm(enumerate(all_prompt)):
            sd = random.randint(0, 2**32)
            image = pipe(prompt, guidance_scale=cfg, seed=sd, eta=eta)[0][0]
            path = os.path.join(new_save_path, "{:05d}_000.png".format(i))
            image.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="model_name_or_path.")
    parser.add_argument(
        "--file",
        default="./data/mscoco.en.1k",
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
        help=
        "Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler-ancest']",
    )
    parser.add_argument("--device", default="gpu", type=str, help="device")
    parser.add_argument("--save_path",
                        default="output/1.5b_ldm/12w.pd",
                        type=str,
                        help="Path to the output file.")
    parser.add_argument("--guidance_scales",
                        default="3 4 5 6 7 8",
                        nargs="+",
                        type=int,
                        help="guidance_scales list.")
    args = parser.parse_args()
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')
    generate_images(model_name_or_path=args.model_name_or_path,
                    file=args.file,
                    save_path=args.save_path,
                    seed=args.seed,
                    scheduler_type=args.scheduler_type,
                    device=args.device)
