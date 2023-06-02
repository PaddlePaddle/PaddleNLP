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
import random

import numpy as np
import paddle
from adapter import DataArguments, Fill50kDataset, GenerateArguments, TextImagePair
from annotator.canny import CannyDetector
from annotator.util import HWC3
from PIL import Image
from tqdm import tqdm

from paddlenlp.trainer import PdArgumentParser
from ppdiffusers import (
    ControlNetModel,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    T2IAdapter,
)

DEFAULT_NEGATIVE_PROMPT = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality"
)


class CannyProcessor:
    """
    canny wrapper.
    """

    def __init__(self, is_output_3d=False):
        self.is_output_3d = is_output_3d
        self.canny_thresh = (100, 200)
        self.apply_canny = CannyDetector()

    def process_data_load(self, image):
        """
        Args:
          image: PIL image.
        Return:
          numpy or tensor. (0 ~ 1)
        """
        image = np.array(image)
        img = HWC3(image)
        H, W, C = img.shape
        # TODO: random thresh.
        detected_map = self.apply_canny(img, *self.canny_thresh)
        if self.is_output_3d:
            detected_map = HWC3(detected_map)
        detected_map = Image.fromarray(detected_map)
        return detected_map

    def process_model_forward(self, image):
        """
        Args:
          tensor (GPU)
        Return:
          tensor (GPU)
        """
        return image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def generate_images(
    use_controlnet=False,
    adapter_model_name_or_path=None,
    sd_model_name_or_path=None,
    batch_size=16,
    test_dataset=None,
    save_path="output",
    guidance_scales=[3, 4, 5, 6, 7, 8],
    num_inference_steps=50,
    scheduler_type="ddim",
    device="gpu",
    max_generation_limits=1000,
    use_text_cond=True,
    use_default_neg_text_cond=True,
    generate_control_image_processor_type=None,
    eta=0.0,
):
    # set pipe
    paddle.set_device(device)
    if use_controlnet:
        controlnet = ControlNetModel.from_pretrained(adapter_model_name_or_path)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_name_or_path, controlnet=controlnet, safety_checker=None
        )
    else:
        adapter = T2IAdapter.from_pretrained(adapter_model_name_or_path)
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            sd_model_name_or_path, adapter=adapter, safety_checker=None
        )
    pipe.set_progress_bar_config(disable=True)

    # set scheduler
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

    # generate
    if generate_control_image_processor_type == "canny":
        if use_controlnet:
            canny_processor = CannyProcessor(is_output_3d=True)
        else:
            canny_processor = CannyProcessor()
    for cfg in guidance_scales:
        set_seed(generate_args.seed)
        new_save_path = os.path.join(save_path, f"cfg_{cfg}")
        os.makedirs(new_save_path, exist_ok=True)
        cond_save_path = os.path.join(save_path, "adapter_cond")
        os.makedirs(cond_save_path, exist_ok=True)
        origin_save_path = os.path.join(save_path, "origin_cond")
        os.makedirs(origin_save_path, exist_ok=True)
        write_file = open(os.path.join(save_path, "caption.txt"), "w")
        i = 0
        for data in tqdm(test_dataset):
            if (
                generate_control_image_processor_type == "canny"
            ):  # Canny mode needs to manually process the control image
                data["adapter_cond"] = canny_processor.process_data_load(data["pixel_values"])
            images = pipe(
                data["input_ids"] if use_text_cond else "",
                negative_prompt=DEFAULT_NEGATIVE_PROMPT if use_default_neg_text_cond else "",
                image=data["adapter_cond"],
                guidance_scale=float(cfg),
                eta=eta,
                num_inference_steps=num_inference_steps,
            )[0]
            data["adapter_cond"].save(os.path.join(cond_save_path, "{:05d}_000.png".format(i)))
            data["pixel_values"].save(os.path.join(origin_save_path, "{:05d}_000.png".format(i)))
            write_file.write("{:05d}_000".format(i) + "\t" + data["input_ids"].strip() + "\n")
            for image in images:
                path = os.path.join(new_save_path, "{:05d}_000.png".format(i))
                image.save(path)
                i += 1
            if i % max_generation_limits == 0:
                break


if __name__ == "__main__":
    parser = PdArgumentParser((DataArguments, GenerateArguments))
    data_args, generate_args = parser.parse_args_into_dataclasses()
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(generate_args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")
    set_seed(generate_args.seed)

    if generate_args.use_dumpy_dataset:
        test_dataset = Fill50kDataset(
            tokenizer=None,
            file_path=generate_args.file,
            do_image_processing=False,
            do_text_processing=False,
        )

    else:
        test_dataset = TextImagePair(
            file_list=generate_args.file,
            size=data_args.resolution,
            num_records=data_args.num_records,
            buffer_size=data_args.buffer_size,
            shuffle_every_n_samples=data_args.shuffle_every_n_samples,
            interpolation="lanczos",
            data_format=generate_args.generate_data_format,
            control_image_processor=None,
            do_image_processing=False,
        )

    generate_images(
        use_controlnet=generate_args.use_controlnet,
        adapter_model_name_or_path=generate_args.adapter_model_name_or_path,
        sd_model_name_or_path=generate_args.sd_model_name_or_path,
        batch_size=generate_args.batch_size,
        test_dataset=test_dataset,
        save_path=generate_args.save_path,
        guidance_scales=generate_args.guidance_scales,
        num_inference_steps=generate_args.num_inference_steps,
        scheduler_type=generate_args.scheduler_type,
        device=generate_args.device,
        max_generation_limits=generate_args.max_generation_limits,
        use_text_cond=generate_args.use_text_cond,
        use_default_neg_text_cond=generate_args.use_default_neg_text_cond,
        generate_control_image_processor_type=generate_args.generate_control_image_processor_type,
    )
