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
import time

# isort: split
import paddle

# isort: split
import fastdeploy as fd
import numpy as np
from tqdm.auto import trange

from paddlenlp.trainer.argparser import strtobool
from ppdiffusers import FastDeployStableDiffusionMegaPipeline
from ppdiffusers.utils import load_image


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="runwayml/stable-diffusion-v1-5@fastdeploy",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument("--inference_steps", type=int, default=50, help="The number of unet inference steps.")
    parser.add_argument("--benchmark_steps", type=int, default=1, help="The number of performance benchmark steps.")
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle_tensorrt",
        # Note(zhoushunjie): Will support 'tensorrt' soon.
        choices=["onnx_runtime", "paddle", "paddlelite", "paddle_tensorrt"],
        help="The inference runtime backend of unet model and text encoder model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        # Note(shentanyue): Will support more devices.
        choices=[
            "cpu",
            "gpu",
            "huawei_ascend_npu",
            "kunlunxin_xpu",
        ],
        help="The inference runtime device of models.",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--task_name",
        type=str,
        default="text2img",
        choices=[
            "text2img",
            "img2img",
            "inpaint",
            "inpaint_legacy",
            "cycle_diffusion",
            "all",
        ],
        help="The task can be one of [text2img, img2img, inpaint, inpaint_legacy, cycle_diffusion, all]. ",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="preconfig-euler-ancestral",
        choices=[
            "pndm",
            "lms",
            "preconfig-lms",
            "euler",
            "euler-ancestral",
            "preconfig-euler-ancestral",
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
    parser.add_argument("--is_sd2_0", type=strtobool, default=False, help="Is sd2_0 model?")

    return parser.parse_args()


def create_ort_runtime(device_id=0):
    option = fd.RuntimeOption()
    option.use_ort_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    return option


def create_paddle_inference_runtime(
    use_trt=False,
    dynamic_shape=None,
    use_fp16=False,
    device_id=0,
    disable_paddle_trt_ops=[],
    disable_paddle_pass=[],
    paddle_stream=None,
    workspace=None,
):
    option = fd.RuntimeOption()
    option.use_paddle_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    if paddle_stream is not None:
        option.set_external_raw_stream(paddle_stream)
    for pass_name in disable_paddle_pass:
        option.paddle_infer_option.delete_pass(pass_name)
    if use_trt:
        option.paddle_infer_option.disable_trt_ops(disable_paddle_trt_ops)
        option.paddle_infer_option.enable_trt = True
        if workspace is not None:
            option.set_trt_max_workspace_size(workspace)
        if use_fp16:
            option.trt_option.enable_fp16 = True
        else:
            # Note(zhoushunjie): These four passes don't support fp32 now.
            # Remove this line of code in future.
            only_fp16_passes = [
                "trt_cross_multihead_matmul_fuse_pass",
                "trt_flash_multihead_matmul_fuse_pass",
                "preln_elementwise_groupnorm_act_pass",
                "elementwise_groupnorm_act_pass",
            ]
            for curr_pass in only_fp16_passes:
                option.paddle_infer_option.delete_pass(curr_pass)

        # Need to enable collect shape
        if dynamic_shape is not None:
            option.paddle_infer_option.collect_trt_shape = True
            for key, shape_dict in dynamic_shape.items():
                option.trt_option.set_shape(
                    key, shape_dict["min_shape"], shape_dict.get("opt_shape", None), shape_dict.get("max_shape", None)
                )
    return option


def create_paddle_lite_runtime(device="cpu", device_id=0):
    option = fd.RuntimeOption()
    option.use_lite_backend()
    if device == "huawei_ascend_npu":
        option.use_ascend()
        option.set_lite_device_names(["huawei_ascend_npu"])
        option.set_lite_context_properties(
            "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS={};HUAWEI_ASCEND_NPU_PRECISION_MODE=allow_mix_precision".format(
                device_id
            )
        )
    elif device == "kunlunxin_xpu":
        # TODO(shentanyue): Add kunlunxin_xpu code
        pass
    else:
        pass
    return option


def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
        paddle_stream = None
    else:
        paddle.set_device(f"gpu:{args.device_id}")
        paddle_stream = paddle.device.cuda.current_stream(args.device_id).cuda_stream

    vae_in_channels = 4
    max_length = 77
    min_image_size = 512
    max_image_size = 512
    max_image_size = max(min_image_size, max_image_size)
    hidden_states = 1024 if args.is_sd2_0 else 768
    unet_in_channels = 9 if args.task_name == "inpaint" else 4

    text_encoder_shape = {
        "input_ids": {
            "min_shape": [1, max_length],
            "max_shape": [1, max_length],
            "opt_shape": [1, max_length],
        }
    }
    vae_encoder_dynamic_shape = {
        "sample": {
            "min_shape": [1, 3, min_image_size, min_image_size],
            "max_shape": [1, 3, max_image_size, max_image_size],
            "opt_shape": [1, 3, min_image_size, min_image_size],
        }
    }
    vae_decoder_dynamic_shape = {
        "latent_sample": {
            "min_shape": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
            "max_shape": [1, vae_in_channels, max_image_size // 8, max_image_size // 8],
            "opt_shape": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
        }
    }
    unet_dynamic_shape = {
        "sample": {
            "min_shape": [1, unet_in_channels, min_image_size // 8, min_image_size // 8],
            "max_shape": [4, unet_in_channels, max_image_size // 8, max_image_size // 8],
            "opt_shape": [2, unet_in_channels, min_image_size // 8, min_image_size // 8],
        },
        "timestep": {
            "min_shape": [1],
            "max_shape": [1],
            "opt_shape": [1],
        },
        "encoder_hidden_states": {
            "min_shape": [1, max_length, hidden_states],
            "max_shape": [4, max_length, hidden_states],
            "opt_shape": [2, max_length, hidden_states],
        },
    }
    # 4. Init runtime
    if args.backend == "onnx_runtime":
        runtime_options = dict(
            text_encoder=create_ort_runtime(device_id=args.device_id),
            vae_encoder=create_ort_runtime(device_id=args.device_id),
            vae_decoder=create_ort_runtime(device_id=args.device_id),
            unet=create_ort_runtime(device_id=args.device_id),
        )
    elif args.backend == "paddlelite":
        runtime_options = dict(
            text_encoder=create_paddle_lite_runtime(device=args.device, device_id=args.device_id),
            vae_encoder=create_paddle_lite_runtime(device=args.device, device_id=args.device_id),
            vae_decoder=create_paddle_lite_runtime(device=args.device, device_id=args.device_id),
            unet=create_paddle_lite_runtime(device=args.device, device_id=args.device_id),
        )
    elif args.backend == "paddle" or args.backend == "paddle_tensorrt":
        args.use_trt = args.backend == "paddle_tensorrt"
        runtime_options = dict(
            text_encoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=text_encoder_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
                disable_paddle_trt_ops=["arg_max", "range", "lookup_table_v2"],
                paddle_stream=paddle_stream,
            ),
            vae_encoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=vae_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
            vae_decoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=vae_decoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
            unet=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=unet_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
        )
    pipe = FastDeployStableDiffusionMegaPipeline.from_pretrained(
        args.model_dir,
        runtime_options=runtime_options,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.change_scheduler(args.scheduler)
    width = args.width
    height = args.height

    if args.task_name in ["text2img", "all"]:
        # text2img
        prompt = "a photo of an astronaut riding a horse on mars"
        time_costs = []
        # warmup
        pipe.text2img(prompt, num_inference_steps=10, height=height, width=width)
        print("==> Test text2img performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            images = pipe.text2img(prompt, num_inference_steps=args.inference_steps, height=height, width=width).images
            latency = time.time() - start
            time_costs += [latency]
            # print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        images[0].save("text2img.png")

    if args.task_name in ["img2img", "all"]:
        # img2img
        img_url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
        )
        init_image = load_image(img_url)
        prompt = "A fantasy landscape, trending on artstation"
        time_costs = []
        # warmup
        pipe.img2img(prompt, image=init_image, num_inference_steps=20, height=height, width=width)
        print("==> Test img2img performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            images = pipe.img2img(
                prompt, image=init_image, num_inference_steps=args.inference_steps, height=height, width=width
            ).images
            latency = time.time() - start
            time_costs += [latency]
            # print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        images[0].save("img2img.png")

    if args.task_name in ["inpaint", "inpaint_legacy", "all"]:
        img_url = (
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
        )
        mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
        init_image = load_image(img_url)
        mask_image = load_image(mask_url)
        prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        time_costs = []
        # warmup
        if args.task_name in ["inpaint_legacy", "all"]:
            call_fn = pipe.inpaint_legacy
        else:
            call_fn = pipe.inpaint
        call_fn(prompt, image=init_image, mask_image=mask_image, num_inference_steps=20, height=height, width=width)
        print("==> Test inpaint_legacy performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            images = call_fn(
                prompt,
                image=init_image,
                mask_image=mask_image,
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
        if args.task_name == "all":
            task_name = "inpaint_legacy"
        else:
            task_name = args.task_name
        images[0].save(f"{task_name}.png")

    if args.task_name in ["cycle_diffusion", "all"]:
        image_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/ride_on_horse.png"
        init_image = load_image(image_url)
        source_prompt = "An astronaut riding a horse"
        prompt = "An astronaut riding an elephant"
        time_costs = []
        # warmup
        pipe.cycle_diffusion(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=10,
            eta=0.1,
            strength=0.8,
            guidance_scale=2,
            source_guidance_scale=1,
            height=height,
            width=width,
        ).images[0]
        print("==> Test cycle diffusion performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            images = pipe.cycle_diffusion(
                prompt=prompt,
                source_prompt=source_prompt,
                image=init_image,
                num_inference_steps=args.inference_steps,
                eta=0.1,
                strength=0.8,
                guidance_scale=2,
                source_guidance_scale=1,
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
        images[0].save("cycle_diffusion.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
