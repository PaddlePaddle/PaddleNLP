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

import math
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import paddle

from ...models import T5FilmDecoder
from ...schedulers import DDPMScheduler
from ...utils import logging, randn_tensor
from ..pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from .continous_encoder import SpectrogramContEncoder
from .notes_encoder import SpectrogramNotesEncoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
TARGET_FEATURE_LENGTH = 256


class SpectrogramDiffusionPipeline(DiffusionPipeline):
    _optional_components = ["melgan"]

    def __init__(
        self,
        notes_encoder: SpectrogramNotesEncoder,
        continuous_encoder: SpectrogramContEncoder,
        decoder: T5FilmDecoder,
        scheduler: DDPMScheduler,
        melgan: (Any),
    ) -> None:
        super().__init__()

        # From MELGAN
        self.min_value = math.log(1e-05)  # Matches MelGAN training.
        self.max_value = 4.0  # Largest value for most examples
        self.n_dims = 128
        self.register_modules(
            notes_encoder=notes_encoder,
            continuous_encoder=continuous_encoder,
            decoder=decoder,
            scheduler=scheduler,
            melgan=melgan,
        )

    def scale_features(self, features, output_range=(-1.0, 1.0), clip=False):
        """Linearly scale features to network outputs range."""
        min_out, max_out = output_range
        if clip:
            features = paddle.clip(x=features, min=self.min_value, max=self.max_value)
        # Scale to [0, 1].
        zero_one = (features - self.min_value) / (self.max_value - self.min_value)
        # Scale to [min_out, max_out].
        return zero_one * (max_out - min_out) + min_out

    def scale_to_features(self, outputs, input_range=(-1.0, 1.0), clip=False):
        """Invert by linearly scaling network outputs to features range."""
        min_out, max_out = input_range
        outputs = paddle.clip(x=outputs, min=min_out, max=max_out) if clip else outputs
        # Scale to [0, 1].
        zero_one = (outputs - min_out) / (max_out - min_out)
        # Scale to [self.min_value, self.max_value].
        return zero_one * (self.max_value - self.min_value) + self.min_value

    def encode(self, input_tokens, continuous_inputs, continuous_mask):
        tokens_mask = input_tokens > 0
        tokens_encoded, tokens_mask = self.notes_encoder(
            encoder_input_tokens=input_tokens, encoder_inputs_mask=tokens_mask
        )
        continuous_encoded, continuous_mask = self.continuous_encoder(
            encoder_inputs=continuous_inputs.cast(self.continuous_encoder.dtype), encoder_inputs_mask=continuous_mask
        )
        return [(tokens_encoded, tokens_mask), (continuous_encoded, continuous_mask)]

    def decode(self, encodings_and_masks, input_tokens, noise_time):
        timesteps = noise_time
        if not paddle.is_tensor(x=timesteps):
            timesteps = paddle.to_tensor(data=[timesteps], dtype="int64", place=input_tokens.place)
        elif paddle.is_tensor(x=timesteps) and len(timesteps.shape) == 0:
            if isinstance(input_tokens.place, paddle.dtype):
                dtype = input_tokens.place
            elif isinstance(input_tokens.place, str) and input_tokens.place not in ["cpu", "cuda", "ipu", "xpu"]:
                dtype = input_tokens.place
            elif isinstance(input_tokens.place, paddle.Tensor):
                dtype = input_tokens.place.dtype
            else:
                dtype = timesteps[None].dtype
            timesteps = timesteps[None].cast(dtype)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * paddle.ones(shape=input_tokens.shape[0], dtype=timesteps.dtype)
        logits = self.decoder(
            encodings_and_masks=encodings_and_masks, decoder_input_tokens=input_tokens, decoder_noise_time=timesteps
        )
        return logits

    @paddle.no_grad()
    def __call__(
        self,
        input_tokens: List[List[int]],
        generator: Optional[paddle.Generator] = None,
        num_inference_steps: int = 100,
        return_dict: bool = True,
        output_type: str = "mel",
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[AudioPipelineOutput, Tuple]:
        if (
            callback_steps is None
            or callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}."
            )
        pred_mel = np.zeros([1, TARGET_FEATURE_LENGTH, self.n_dims], dtype=np.float32)
        full_pred_mel = np.zeros([1, 0, self.n_dims], np.float32)
        ones = paddle.ones(shape=(1, TARGET_FEATURE_LENGTH), dtype=bool)
        for i, encoder_input_tokens in enumerate(input_tokens):
            if i == 0:
                encoder_continuous_inputs = paddle.to_tensor(data=pred_mel[:1].copy()).cast(self.decoder.dtype)
                # The first chunk has no previous context.
                encoder_continuous_mask = paddle.zeros(shape=(1, TARGET_FEATURE_LENGTH), dtype=bool)
            else:
                # The full song pipeline does not feed in a context feature, so the mask
                # will be all 0s after the feature converter. Because we know we're
                # feeding in a full context chunk from the previous prediction, set it
                # to all 1s.
                encoder_continuous_mask = ones
            encoder_continuous_inputs = self.scale_features(
                encoder_continuous_inputs, output_range=[-1.0, 1.0], clip=True
            )
            encodings_and_masks = self.encode(
                input_tokens=paddle.to_tensor(data=[encoder_input_tokens], dtype="int32"),
                continuous_inputs=encoder_continuous_inputs,
                continuous_mask=encoder_continuous_mask,
            )

            # Sample encoder_continuous_inputs shaped gaussian noise to begin loop
            x = randn_tensor(shape=encoder_continuous_inputs.shape, generator=generator, dtype=self.decoder.dtype)

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            # Denoising diffusion loop
            for j, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                output = self.decode(
                    encodings_and_masks=encodings_and_masks,
                    input_tokens=x,
                    noise_time=t / self.scheduler.config.num_train_timesteps,
                )

                # Compute previous output: x_t -> x_t-1
                x = self.scheduler.step(output, t, x, generator=generator).prev_sample
            mel = self.scale_to_features(x, input_range=[-1.0, 1.0])
            encoder_continuous_inputs = mel[:1]
            pred_mel = mel.cpu().astype(dtype="float32").numpy()
            full_pred_mel = np.concatenate([full_pred_mel, pred_mel[:1]], axis=1)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, full_pred_mel)
            logger.info("Generated segment", i)
        if output_type == "numpy":
            output = self.melgan(input_features=full_pred_mel.astype(np.float32))[0]
        else:
            output = full_pred_mel
        if not return_dict:
            return (output,)
        return AudioPipelineOutput(audios=output)
