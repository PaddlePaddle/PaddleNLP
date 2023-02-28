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
import time
from io import BytesIO

import fastdeploy as fd
import numpy as np
import paddle
import requests
from fastdeploy import ModelFormat
from PIL import Image

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.transformers import CLIPTokenizer
from ppdiffusers import (
    DDIMScheduler,
    FastDeployCycleDiffusionPipeline,
    FastDeployRuntimeModel,
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
    parser.add_argument("--inference_steps", type=int, default=100, help="The number of unet inference steps.")
    parser.add_argument("--benchmark_steps", type=int, default=1, help="The number of performance benchmark steps.")
    parser.add_argument(
        "--image_path", default="horse_to_elephant.png", help="The model directory of diffusion_model."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        # Note(zhoushunjie): Will support 'tensorrt', 'paddle-tensorrt' soon.
        choices=["onnx_runtime", "paddle", "paddle-tensorrt", "tensorrt", "paddlelite"],
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
    parser.add_argument("--use_fp16", type=strtobool, default=False, help="Wheter to use FP16 mode")
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
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
        if use_fp16:
            option.trt_option.enable_fp16 = True
        cache_file = os.path.join(model_dir, model_prefix, "inference.trt")
        option.trt_option.serialize_file = cache_file
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
        option.paddle_lite_option.nnadapter_model_cache_dir = os.path.join(model_dir, model_prefix)
        option.paddle_lite_option.nnadapter_context_properties = (
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
    option.trt_option.enable_fp16 = True
    option.trt_option.max_workspace_size = workspace
    if dynamic_shape is not None:
        for key, shape_dict in dynamic_shape.items():
            option.trt_option.set_shape(
                key, shape_dict["min_shape"], shape_dict.get("opt_shape", None), shape_dict.get("max_shape", None)
            )
    if model_format == "paddle":
        model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
        params_file = os.path.join(model_dir, model_prefix, "inference.pdiparams")
        option.set_model_path(model_file, params_file)
    else:
        onnx_file = os.path.join(model_dir, model_prefix, "inference.onnx")
        option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    cache_file = os.path.join(model_dir, model_prefix, "inference.trt")
    option.trt_option.serialize_file = cache_file
    return fd.Runtime(option)


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
    scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

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
    text_encoder_shape = {
        "input_ids": {
            "min_shape": [1, 77],
            "max_shape": [2, 77],
            "opt_shape": [1, 77],
        }
    }
    unet_dynamic_shape = {
        "sample": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [4, 4, 64, 64],
            "opt_shape": [4, 4, 64, 64],
        },
        "timestep": {
            "min_shape": [1],
            "max_shape": [1],
            "opt_shape": [1],
        },
        "encoder_hidden_states": {
            "min_shape": [1, 77, 768],
            "max_shape": [4, 77, 768],
            "opt_shape": [4, 77, 768],
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
    elif args.backend == "paddle" or args.backend == "paddle-tensorrt":
        use_trt = True if args.backend == "paddle-tensorrt" else False
        text_encoder_runtime = create_paddle_inference_runtime(
            args.model_dir,
            args.text_encoder_model_prefix,
            use_trt,
            text_encoder_shape,
            use_fp16=args.use_fp16,
            device_id=device_id,
            disable_paddle_trt_ops=["arg_max", "range", "lookup_table_v2"],
            paddle_stream=paddle_stream,
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

    pipe = FastDeployCycleDiffusionPipeline(
        vae_encoder=FastDeployRuntimeModel(model=vae_encoder_runtime),
        vae_decoder=FastDeployRuntimeModel(model=vae_decoder_runtime),
        text_encoder=FastDeployRuntimeModel(model=text_encoder_runtime),
        tokenizer=tokenizer,
        unet=FastDeployRuntimeModel(model=unet_runtime),
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )

    # 5. Download an initial image
    url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/An%20astronaut%20riding%20a%20horse.png"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((512, 512))
    init_image.save("horse.png")

    # 6. Specify a prompt
    source_prompt = "An astronaut riding a horse"
    prompt = "An astronaut riding an elephant"

    # 7. Call the pipeline
    # Warm up
    pipe(
        prompt=prompt,
        source_prompt=source_prompt,
        image=init_image,
        num_inference_steps=10,
        eta=0.1,
        strength=0.8,
        guidance_scale=2,
        source_guidance_scale=1,
    )
    time_costs = []
    print(f"Run the cycle diffusion pipeline {args.benchmark_steps} times to test the performance.")
    for step in range(args.benchmark_steps):
        start = time.time()
        image = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=args.inference_steps,
            eta=0.1,
            strength=0.8,
            guidance_scale=2,
            source_guidance_scale=1,
        ).images[0]
        latency = time.time() - start
        time_costs += [latency]
        print(f"No {step:3d} time cost: {latency:2f} s")
    print(
        f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
        f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
    )
    image.save(f"{args.image_path}")
    print(f"Image saved in {args.image_path}!")
