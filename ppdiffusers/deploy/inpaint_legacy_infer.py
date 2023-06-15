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
import time
from io import BytesIO

# isort: split
import paddle

# isort: split
import fastdeploy as fd
import numpy as np
import PIL
import requests
from fastdeploy import ModelFormat

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import CLIPTokenizer
from ppdiffusers import (
    EulerAncestralDiscreteScheduler,
    FastDeployRuntimeModel,
    FastDeployStableDiffusionInpaintPipelineLegacy,
    PNDMScheduler,
)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", default="paddle_diffusion_model", help="The model directory of diffusion_model."
    )
    parser.add_argument("--model_format", default="paddle", choices=["paddle", "onnx"], help="The model format.")
    parser.add_argument("--unet_model_prefix", default="unet", help="The file prefix of unet model.")
    parser.add_argument(
        "--vae_decoder_model_prefix", default="vae_decoder", help="The file prefix of vae decoder model."
    )
    parser.add_argument(
        "--vae_encoder_model_prefix", default="vae_encoder", help="The file prefix of vae encoder model."
    )
    parser.add_argument(
        "--text_encoder_model_prefix", default="text_encoder", help="The file prefix of text_encoder model."
    )
    parser.add_argument("--inference_steps", type=int, default=50, help="The number of unet inference steps.")
    parser.add_argument("--benchmark_steps", type=int, default=1, help="The number of performance benchmark steps.")
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        # Note(zhoushunjie): Will support 'tensorrt', 'paddle_tensorrt' soon.
        choices=["onnx_runtime", "paddle", "paddle_tensorrt", "tensorrt", "paddlelite"],
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
    parser.add_argument("--image_path", default="cat_on_bench_new.png", help="The model directory of diffusion_model.")
    parser.add_argument("--use_fp16", type=strtobool, default=False, help="Wheter to use FP16 mode")
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler_ancestral",
        choices=["pndm", "euler_ancestral"],
        help="The scheduler type of stable diffusion.",
    )
    return parser.parse_args()


def create_ort_runtime(model_dir, model_prefix, model_format, device_id=0):
    option = fd.RuntimeOption()
    option.use_ort_backend()
    option.use_gpu(device_id)
    if model_format == "paddle":
        model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
        params_file = os.path.join(model_dir, model_prefix, "inference.pdiparams")
        option.set_model_path(model_file, params_file)
    else:
        onnx_file = os.path.join(model_dir, model_prefix, "inference.onnx")
        option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    return fd.Runtime(option)


def create_paddle_inference_runtime(
    model_dir,
    model_prefix,
    use_trt=False,
    dynamic_shape=None,
    use_fp16=False,
    device_id=0,
    disable_paddle_trt_ops=[],
    disable_paddle_pass=[],
    paddle_stream=None,
):
    option = fd.RuntimeOption()
    option.use_paddle_infer_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    # (TODO, junnyu) remove use_trt
    if use_trt and paddle_stream is not None:
        option.set_external_raw_stream(paddle_stream)
    for pass_name in disable_paddle_pass:
        option.paddle_infer_option.delete_pass(pass_name)
    if use_trt:
        option.paddle_infer_option.disable_trt_ops(disable_paddle_trt_ops)
        option.paddle_infer_option.enable_trt = True
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
        cache_file = os.path.join(model_dir, model_prefix, "_opt_cache/")
        option.set_trt_cache_file(cache_file)
        # Need to enable collect shape for ernie
        if dynamic_shape is not None:
            option.paddle_infer_option.collect_trt_shape = True
            for key, shape_dict in dynamic_shape.items():
                option.trt_option.set_shape(
                    key, shape_dict["min_shape"], shape_dict.get("opt_shape", None), shape_dict.get("max_shape", None)
                )
    model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
    params_file = os.path.join(model_dir, model_prefix, "inference.pdiparams")
    option.set_model_path(model_file, params_file)
    return fd.Runtime(option)


def create_paddle_lite_runtime(model_dir, model_prefix, device="cpu", device_id=0):
    option = fd.RuntimeOption()
    option.use_lite_backend()
    if device == "huawei_ascend_npu":
        option.use_ascend()
        option.set_lite_device_names(["huawei_ascend_npu"])
        option.set_lite_model_cache_dir(os.path.join(model_dir, model_prefix))
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
    model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
    params_file = os.path.join(model_dir, model_prefix, "inference.pdiparams")
    option.set_model_path(model_file, params_file)
    return fd.Runtime(option)


def create_trt_runtime(model_dir, model_prefix, model_format, workspace=(1 << 31), dynamic_shape=None, device_id=0):
    option = fd.RuntimeOption()
    option.use_trt_backend()
    option.use_gpu(device_id)
    option.enable_trt_fp16()
    option.set_trt_max_workspace_size(workspace)
    if dynamic_shape is not None:
        for key, shape_dict in dynamic_shape.items():
            option.set_trt_input_shape(
                key,
                min_shape=shape_dict["min_shape"],
                opt_shape=shape_dict.get("opt_shape", None),
                max_shape=shape_dict.get("max_shape", None),
            )
    if model_format == "paddle":
        model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
        params_file = os.path.join(model_dir, model_prefix, "inference.pdiparams")
        option.set_model_path(model_file, params_file)
    else:
        onnx_file = os.path.join(model_dir, model_prefix, "inference.onnx")
        option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    cache_file = os.path.join(model_dir, model_prefix, "inference.trt")
    option.set_trt_cache_file(cache_file)
    return fd.Runtime(option)


def get_scheduler(args):
    if args.scheduler == "pndm":
        scheduler = PNDMScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            skip_prk_steps=True,
        )
    elif args.scheduler == "euler_ancestral":
        scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    else:
        raise ValueError(f"Scheduler '{args.scheduler}' is not supportted right now.")
    return scheduler


if __name__ == "__main__":
    args = parse_arguments()
    # 0. Init device id
    device_id = args.device_id
    if args.device == "cpu":
        device_id = -1
        paddle.set_device("cpu")
        paddle_stream = None
    else:
        paddle.set_device(f"gpu:{device_id}")
        paddle_stream = paddle.device.cuda.current_stream(device_id).cuda_stream
    # 1. Init scheduler
    scheduler = get_scheduler(args)

    # 2. Init tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))

    # 3. Set dynamic shape for trt backend
    vae_decoder_dynamic_shape = {
        "latent_sample": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [2, 4, 64, 64],
            "opt_shape": [2, 4, 64, 64],
        }
    }

    vae_encoder_dynamic_shape = {
        "sample": {
            "min_shape": [1, 3, 512, 512],
            "max_shape": [2, 3, 512, 512],
            "opt_shape": [2, 3, 512, 512],
        }
    }

    unet_dynamic_shape = {
        "sample": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [2, 4, 64, 64],
            "opt_shape": [2, 4, 64, 64],
        },
        "timestep": {
            "min_shape": [1],
            "max_shape": [1],
            "opt_shape": [1],
        },
        "encoder_hidden_states": {
            "min_shape": [1, 77, 768],
            "max_shape": [2, 77, 768],
            "opt_shape": [2, 77, 768],
        },
    }

    # 4. Init runtime
    if args.backend == "onnx_runtime":
        text_encoder_runtime = create_ort_runtime(
            args.model_dir, args.text_encoder_model_prefix, args.model_format, device_id=device_id
        )
        vae_decoder_runtime = create_ort_runtime(
            args.model_dir, args.vae_decoder_model_prefix, args.model_format, device_id=device_id
        )
        vae_encoder_runtime = create_ort_runtime(
            args.model_dir, args.vae_encoder_model_prefix, args.model_format, device_id=device_id
        )
        start = time.time()
        unet_runtime = create_ort_runtime(
            args.model_dir, args.unet_model_prefix, args.model_format, device_id=device_id
        )
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    elif args.backend == "paddle" or args.backend == "paddle_tensorrt":
        use_trt = True if args.backend == "paddle_tensorrt" else False
        # Note(zhoushunjie): Will change to paddle runtime later
        text_encoder_runtime = create_ort_runtime(
            args.model_dir, args.text_encoder_model_prefix, args.model_format, device_id=device_id
        )
        vae_decoder_runtime = create_paddle_inference_runtime(
            args.model_dir,
            args.vae_decoder_model_prefix,
            use_trt,
            vae_decoder_dynamic_shape,
            use_fp16=args.use_fp16,
            device_id=device_id,
            paddle_stream=paddle_stream,
        )
        vae_encoder_runtime = create_paddle_inference_runtime(
            args.model_dir,
            args.vae_encoder_model_prefix,
            use_trt,
            vae_encoder_dynamic_shape,
            use_fp16=args.use_fp16,
            device_id=device_id,
            paddle_stream=paddle_stream,
        )
        start = time.time()
        unet_runtime = create_paddle_inference_runtime(
            args.model_dir,
            args.unet_model_prefix,
            use_trt,
            unet_dynamic_shape,
            use_fp16=args.use_fp16,
            device_id=device_id,
            paddle_stream=paddle_stream,
        )
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    elif args.backend == "tensorrt":
        text_encoder_runtime = create_ort_runtime(args.model_dir, args.text_encoder_model_prefix, args.model_format)
        vae_decoder_runtime = create_trt_runtime(
            args.model_dir,
            args.vae_decoder_model_prefix,
            args.model_format,
            workspace=(1 << 30),
            dynamic_shape=vae_decoder_dynamic_shape,
            device_id=device_id,
        )
        vae_encoder_runtime = create_trt_runtime(
            args.model_dir,
            args.vae_encoder_model_prefix,
            args.model_format,
            workspace=(1 << 30),
            dynamic_shape=vae_encoder_dynamic_shape,
            device_id=device_id,
        )
        start = time.time()
        unet_runtime = create_trt_runtime(
            args.model_dir,
            args.unet_model_prefix,
            args.model_format,
            dynamic_shape=unet_dynamic_shape,
            device_id=device_id,
        )
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    elif args.backend == "paddlelite":
        text_encoder_runtime = create_paddle_lite_runtime(
            args.model_dir, args.text_encoder_model_prefix, device=args.device, device_id=device_id
        )
        vae_decoder_runtime = create_paddle_lite_runtime(
            args.model_dir, args.vae_decoder_model_prefix, device=args.device, device_id=device_id
        )
        vae_encoder_runtime = create_paddle_lite_runtime(
            args.model_dir, args.vae_encoder_model_prefix, device=args.device, device_id=device_id
        )
        start = time.time()
        unet_runtime = create_paddle_lite_runtime(
            args.model_dir, args.unet_model_prefix, device=args.device, device_id=device_id
        )
        print(f"Spend {time.time() - start : .2f} s to load unet model.")

    pipe = FastDeployStableDiffusionInpaintPipelineLegacy(
        vae_encoder=FastDeployRuntimeModel(model=vae_encoder_runtime),
        vae_decoder=FastDeployRuntimeModel(model=vae_decoder_runtime),
        text_encoder=FastDeployRuntimeModel(model=text_encoder_runtime),
        tokenizer=tokenizer,
        unet=FastDeployRuntimeModel(model=unet_runtime),
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )

    # Download the init image

    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
    mask_url = (
        "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
    )

    init_image = download_image(img_url).resize((512, 512))
    mask_image = download_image(mask_url).resize((512, 512))

    prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    # Warm up
    pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=10)

    time_costs = []
    print(f"Run the stable diffusion inpaint legacy pipeline {args.benchmark_steps} times to test the performance.")

    for step in range(args.benchmark_steps):
        start = time.time()
        images = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=args.inference_steps,
        ).images
        latency = time.time() - start
        time_costs += [latency]
        print(f"No {step:3d} time cost: {latency:2f} s")
    print(
        f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
        f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
    )

    images[0].save(args.image_path)
    print(f"Image saved in {args.image_path}!")
