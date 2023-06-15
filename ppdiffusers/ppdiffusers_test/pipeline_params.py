# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

# These are canonical sets of parameters for different types of pipelines.
# They are set on subclasses of `PipelineTesterMixin` as `params` and
# `batch_params`.
#
# If a pipeline's set of arguments has minor changes from one of the common sets
# of arguments, do not make modifications to the existing common sets of arguments.
# I.e. a text to image pipeline with non-configurable height and width arguments
# should set its attribute as `params = TEXT_TO_IMAGE_PARAMS - {'height', 'width'}`.

TEXT_TO_IMAGE_PARAMS = frozenset(
    [
        "prompt",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
    ]
)

TEXT_TO_IMAGE_BATCH_PARAMS = frozenset(["prompt", "negative_prompt"])

IMAGE_VARIATION_PARAMS = frozenset(
    [
        "image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_VARIATION_BATCH_PARAMS = frozenset(["image"])

TEXT_GUIDED_IMAGE_VARIATION_PARAMS = frozenset(
    [
        "prompt",
        "image",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]
)

TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS = frozenset(["prompt", "image", "negative_prompt"])

TEXT_GUIDED_IMAGE_INPAINTING_PARAMS = frozenset(
    [
        # Text guided image variation with an image mask
        "prompt",
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]
)

TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["prompt", "image", "mask_image", "negative_prompt"])

IMAGE_INPAINTING_PARAMS = frozenset(
    [
        # image variation with an image mask
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["image", "mask_image"])

IMAGE_GUIDED_IMAGE_INPAINTING_PARAMS = frozenset(
    [
        "example_image",
        "image",
        "mask_image",
        "height",
        "width",
        "guidance_scale",
    ]
)

IMAGE_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS = frozenset(["example_image", "image", "mask_image"])

CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS = frozenset(["class_labels"])

CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS = frozenset(["class_labels"])

UNCONDITIONAL_IMAGE_GENERATION_PARAMS = frozenset(["batch_size"])

UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS = frozenset([])

UNCONDITIONAL_AUDIO_GENERATION_PARAMS = frozenset(["batch_size"])

UNCONDITIONAL_AUDIO_GENERATION_BATCH_PARAMS = frozenset([])
