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

import os

import paddle
from adapter import DataArguments, GenerateArguments, TextImagePair

from paddlenlp.trainer import PdArgumentParser
from ppdiffusers import (
    Adapter,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionAdapterPipeline,
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
    adapter_model_name_or_path,
    sd_model_name_or_path,
    batch_size=16,
    test_dataset=None,
    save_path="output",
    seed=4096,
    scheduler_type="ddim",
    eta=0.0,
    num_inference_steps=50,
    guidance_scales=[3, 4, 5, 6, 7, 8],
    height=256,
    width=256,
    device="gpu",
):
    paddle.set_device(device)
    adapter = Adapter.from_pretrained(adapter_model_name_or_path)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(sd_model_name_or_path, adapter=adapter, safety_checker=None)
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

    for cfg in guidance_scales:
        new_save_path = os.path.join(save_path, f"cfg_{cfg}")
        os.makedirs(new_save_path, exist_ok=True)
        cond_save_path = os.path.join(save_path, "adapter_cond")
        os.makedirs(cond_save_path, exist_ok=True)
        i = 0
        for data in test_dataset:
            paddle.seed(seed)
            images = pipe(
                data["input_ids"],
                image=data["adapter_cond"],
                guidance_scale=float(cfg),
                eta=eta,
                num_inference_steps=num_inference_steps,
            )[0]
            for image in images:
                path = os.path.join(new_save_path, "{:05d}_000.png".format(i))
                image.save(path)
                i += 1
            data["adapter_cond"].save(os.path.join(cond_save_path, "{:05d}_000.png".format(i)))
            if i % 10 == 0:
                break


def collate_fn(examples):
    pixel_values = paddle.stack([paddle.to_tensor(example["pixel_values"]) for example in examples])
    input_ids = paddle.stack([paddle.to_tensor(example["input_ids"]) for example in examples])
    adapter_cond = paddle.stack([paddle.to_tensor(example["adapter_cond"]) for example in examples])

    batch = {"input_ids": input_ids, "pixel_values": pixel_values, "adapter_cond": adapter_cond}
    return batch


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    parser = PdArgumentParser((DataArguments, GenerateArguments))
    data_args, generate_args = parser.parse_args_into_dataclasses()
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(generate_args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")

    test_dataset = TextImagePair(
        file_list=generate_args.file,
        size=data_args.resolution,
        num_records=data_args.num_records,
        buffer_size=data_args.buffer_size,
        shuffle_every_n_samples=data_args.shuffle_every_n_samples,
        interpolation="lanczos",
        data_format="img2img",
    )
    generate_images(
        adapter_model_name_or_path=generate_args.adapter_model_name_or_path,
        sd_model_name_or_path=generate_args.sd_model_name_or_path,
        batch_size=generate_args.batch_size,
        test_dataset=test_dataset,
        save_path=generate_args.save_path,
        seed=generate_args.seed,
        guidance_scales=[3, 5],
        num_inference_steps=generate_args.num_inference_steps,
        scheduler_type=generate_args.scheduler_type,
        height=generate_args.height,
        width=generate_args.width,
        device=generate_args.device,
    )
