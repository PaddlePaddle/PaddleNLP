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

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.transformers import (
    CLIPPretrainedModel,
    CLIPVisionConfig,
    CLIPVisionModel,
)

from ...utils import logging

logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = F.normalize(image_embeds)
    normalized_text_embeds = F.normalize(text_embeds)
    return paddle.matmul(normalized_image_embeds, normalized_text_embeds, transpose_y=True)


class StableDiffusionSafetyChecker(CLIPPretrainedModel):
    config_class = CLIPVisionConfig

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)

        self.clip = CLIPVisionModel(config)
        self.vision_projection = paddle.create_parameter(
            (config.hidden_size, config.projection_dim), dtype=paddle.get_default_dtype()
        )

        self.register_buffer("concept_embeds", paddle.ones([17, config.projection_dim]))
        self.register_buffer("special_care_embeds", paddle.ones([3, config.projection_dim]))

        self.register_buffer("concept_embeds_weights", paddle.ones([17]))
        self.register_buffer("special_care_embeds_weights", paddle.ones([3]))

    @paddle.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.clip(clip_input)[1]  # pooled_output
        image_embeds = paddle.matmul(pooled_output, self.vision_projection)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).astype("float32").numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).astype("float32").numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                images[idx] = np.zeros(images[idx].shape)  # black image

        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts

    def forward_fastdeploy(self, clip_input: paddle.Tensor, images: paddle.Tensor):
        pooled_output = self.clip(clip_input)[1]  # pooled_output
        image_embeds = paddle.matmul(pooled_output, self.vision_projection)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # increase this value to create a stronger `nsfw` filter
        # at the cost of increasing the possibility of filtering benign images
        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # special_scores = special_scores.round(decimals=3)
        special_care = paddle.any(special_scores > 0, axis=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand([-1, cos_dist.shape[1]])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # concept_scores = concept_scores.round(decimals=3)
        has_nsfw_concepts = paddle.any(concept_scores > 0, axis=1)

        images[has_nsfw_concepts] = 0.0  # black image

        return images, has_nsfw_concepts
