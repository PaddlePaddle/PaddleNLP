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

import gc
import unittest

import numpy as np
import paddle

from ppdiffusers import DDPMScheduler, MidiProcessor, SpectrogramDiffusionPipeline
from ppdiffusers.pipelines.spectrogram_diffusion import (
    SpectrogramContEncoder,
    SpectrogramNotesEncoder,
    T5FilmDecoder,
)
from ppdiffusers.training_utils import enable_full_determinism
from ppdiffusers.utils import require_paddle_gpu, slow

from ..pipeline_params import (
    TOKENS_TO_AUDIO_GENERATION_BATCH_PARAMS,
    TOKENS_TO_AUDIO_GENERATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin

enable_full_determinism(42)


MIDI_FILE = "./tests/fixtures/elise_format0.mid"


# The note-seq package throws an error on import because the default installed version of Ipython
# is not compatible with python 3.8 which we run in the CI.
# https://github.com/huggingface/diffusers/actions/runs/4830121056/jobs/8605954838#step:7:98
# @unittest.skip("The note-seq package currently throws an error on import")
class SpectrogramDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = SpectrogramDiffusionPipeline
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "callback",
        "latents",
        "callback_steps",
        "output_type",
        "num_images_per_prompt",
    }
    test_attention_slicing = False
    test_xformers_attention = False
    test_cpu_offload = False
    batch_params = TOKENS_TO_AUDIO_GENERATION_PARAMS
    params = TOKENS_TO_AUDIO_GENERATION_BATCH_PARAMS

    def get_dummy_components(self):
        paddle.seed(0)
        notes_encoder = SpectrogramNotesEncoder(
            max_length=2048,
            vocab_size=1536,
            d_model=768,
            dropout_rate=0.1,
            num_layers=1,
            num_heads=1,
            d_kv=4,
            d_ff=2048,
            feed_forward_proj="gated-gelu",
        )
        notes_encoder.eval()
        paddle.seed(0)
        continuous_encoder = SpectrogramContEncoder(
            input_dims=128,
            targets_context_length=256,
            d_model=768,
            dropout_rate=0.1,
            num_layers=1,
            num_heads=1,
            d_kv=4,
            d_ff=2048,
            feed_forward_proj="gated-gelu",
        )
        continuous_encoder.eval()

        paddle.seed(0)
        decoder = T5FilmDecoder(
            input_dims=128,
            targets_length=256,
            max_decoder_noise_time=20000.0,
            d_model=768,
            num_layers=1,
            num_heads=1,
            d_kv=4,
            d_ff=2048,
            dropout_rate=0.1,
        )
        decoder.eval()

        scheduler = DDPMScheduler()

        components = {
            "notes_encoder": notes_encoder,
            "continuous_encoder": continuous_encoder,
            "decoder": decoder,
            "scheduler": scheduler,
            "melgan": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):

        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "input_tokens": [
                [1134, 90, 1135, 1133, 1080, 112, 1132, 1080, 1133, 1079, 133, 1132, 1079, 1133, 1] + [0] * 2033
            ],
            "generator": generator,
            "num_inference_steps": 4,
            "output_type": "mel",
        }
        return inputs

    def test_spectrogram_diffusion(self):
        components = self.get_dummy_components()
        pipe = SpectrogramDiffusionPipeline(**components)

        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)
        mel = output.audios

        mel_slice = mel[0, -3:, -3:]

        assert mel_slice.shape == (3, 3)
        expected_slice = np.array(
            [-11.46511, 4.0, -8.506372, -11.512925, -11.512925, -10.417862, -8.077912, 3.7985802, 4.0]
        )
        assert np.abs(mel_slice.flatten() - expected_slice).max() < 1e-2

    def test_save_load_local(self):
        return super().test_save_load_local()

    def test_dict_tuple_outputs_equivalent(self):
        return super().test_dict_tuple_outputs_equivalent()

    def test_save_load_optional_components(self):
        return super().test_save_load_optional_components()

    def test_attention_slicing_forward_pass(self):
        return super().test_attention_slicing_forward_pass()

    def test_inference_batch_single_identical(self):
        pass

    def test_inference_batch_consistent(self):
        pass

    def test_progress_bar(self):
        return super().test_progress_bar()


@slow
@require_paddle_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_callback(self):
        # TODO - test that pipeline can decode tokens in a callback
        # so that music can be played live
        pipe = SpectrogramDiffusionPipeline.from_pretrained("google/music-spectrogram-diffusion")
        melgan = pipe.melgan
        pipe.melgan = None

        pipe.set_progress_bar_config(disable=None)

        def callback(step, mel_output):
            # decode mel to audio
            audio = melgan(input_features=mel_output.astype(np.float32))[0]
            assert len(audio[0]) == 81920 * (step + 1)
            # simulate that audio is played
            return audio

        processor = MidiProcessor()
        input_tokens = processor(MIDI_FILE)

        input_tokens = input_tokens[:3]
        generator = paddle.Generator().manual_seed(0)
        pipe(input_tokens, num_inference_steps=5, generator=generator, callback=callback, output_type="mel")

    def test_spectrogram_fast(self):

        pipe = SpectrogramDiffusionPipeline.from_pretrained("google/music-spectrogram-diffusion")
        pipe.set_progress_bar_config(disable=None)
        processor = MidiProcessor()

        input_tokens = processor(MIDI_FILE)
        # just run two denoising loops
        input_tokens = input_tokens[:2]

        generator = paddle.Generator().manual_seed(0)
        output = pipe(input_tokens, num_inference_steps=2, generator=generator)

        audio = output.audios[0]

        assert abs(np.abs(audio).sum() - 3815.163) < 1e-1

    def test_spectrogram(self):

        pipe = SpectrogramDiffusionPipeline.from_pretrained("google/music-spectrogram-diffusion")
        pipe.set_progress_bar_config(disable=None)

        processor = MidiProcessor()

        input_tokens = processor(MIDI_FILE)

        # just run 4 denoising loops
        input_tokens = input_tokens[:4]

        generator = paddle.Generator().manual_seed(0)
        output = pipe(input_tokens, num_inference_steps=100, generator=generator)

        audio = output.audios[0]
        assert abs(np.abs(audio).sum() - 14418.089) < 5e-2
