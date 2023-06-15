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
import numpy as np
import paddle

from ..taskflow import Taskflow
from ..transformers import (
    AutoModelForCausalLM,
    AutoModelForConditionalGeneration,
    AutoTokenizer,
)

__all__ = [
    "SentenceGenerate",
    "SentenceSummarize",
    "SentenceBackTranslate",
    "SentenceBackTranslateAPI",
    "SentenceContinue",
]


class SentenceGenerate:
    """
    SentenceGenerate is a sentence-level data augmentation strategy
    that generates simialr sentences according to the input sequence.
    The strattegy first generates several sentences, and then chooses
    the top n simialr sentences by the model.

    Args:
        model_name (str):
            Model parameter name for generation task.
        create_n (int):
            Number of augmented sequences.
        generate_n (int):
            Number of generated sequences.
        max_length (int):
            The max length of the prediction.
        top_p (float): The cumulative probability for
            top-p-filtering in the "sampling" strategy. The value should
            satisfy 0 <= top_p < 1. Default to 0.95.
    """

    def __init__(
        self, model_name="roformer-chinese-sim-char-base", create_n=1, generate_n=5, max_length=128, top_p=0.95
    ):
        self.model_name = model_name
        self.create_n = create_n
        self.generate_n = generate_n
        self.max_length = max_length
        self.top_p = top_p

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def augment(self, sequences):
        """
        Apply augmentation strategy on input sequences.

            Args:
            sequences (str or list(str)):
                Input sequence or list of input sequences.

        """
        if isinstance(sequences, str):
            sequences = [sequences]
        augmented_sequences = []
        for sequence in sequences:
            augmented_sequences.append(self._generate_similar_sentence(sequence, self.model, self.tokenizer))
        return augmented_sequences

    @paddle.no_grad()
    def _generate_similar_sentence(self, sequence, model, tokenizer):
        """Generates generate_n similar sentences from the provided sequence, and chooose the best create_n similar sentences."""

        # Generate generate_n similar sentences
        generated_sequences = [sequence]
        tokenized_input = tokenizer(sequence, return_tensors="pd", padding=True)
        decoded_outputs = tokenizer.batch_decode(
            model.generate(
                **tokenized_input,
                num_return_sequences=self.generate_n,
                top_p=self.top_p,
                decode_strategy="sampling",
                max_length=self.max_length,
            )[0],
            skip_special_tokens=True,
        )
        for decoded_output in decoded_outputs:
            s = decoded_output.replace(" ", "").replace(sequence, "")
            if s not in generated_sequences and len(s) > 0:
                generated_sequences.append(s)
        tokenized_output = tokenizer(generated_sequences, return_tensors="pd", padding=True)

        # Choose best create_n similar sentences
        tokenized_output = tokenizer(generated_sequences, return_tensors="pd", padding=True)
        Z = model.roformer(**tokenized_output)[1].cpu().numpy()
        Z /= (Z**2).sum(axis=1, keepdims=True) ** 0.5

        return [generated_sequences[i + 1] for i in np.dot(Z[1:], -Z[0]).argsort()[: self.create_n]]


class SentenceSummarize:
    """
    SentenceSummarize is a sentence-level data augmentation strategy
    that summarizes the input sequence.

    Args:
        create_n (int):
            Number of augmented sequences.
        max_length (int):
            The max length of the summarization.
        batch_size(int):
            The sample number of a mini-batch.
        top_k (int): The number of highest probability tokens to
            keep for top-k-filtering in the "sampling" strategy. Default to
            0, which means no effect.
        top_p (float): The cumulative probability for
            top-p-filtering in the "sampling" strategy. The value should
            satisfy 0 <= top_p < 1. Default to 1.0, which means no
            effect.
        temperature (float): The value used to module the next
            token probabilities in the "sampling" strategy. Default to 1.0,
            which means no effect.
        use_fp16_decoding: (bool): Whether to use fp16 for decoding.
            Only works when faster entry is avalible. Default to False.
        kwargs (dict): Additional keyword arguments refer to ..taskflow.text_summarization.TextSummarization
    """

    def __init__(
        self,
        create_n=1,
        max_length=128,
        batch_size=1,
        top_k=5,
        top_p=1.0,
        temperature=1.0,
        use_fp16_decoding=False,
        **kwargs
    ):

        kwargs.setdefault("num_return_sequences", create_n)
        kwargs.setdefault("num_beams", create_n * 4)
        kwargs.setdefault("max_length", max_length)
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("top_k", top_k)
        kwargs.setdefault("top_p", top_p)
        kwargs.setdefault("temperature", temperature)
        kwargs.setdefault("use_fp16_decoding", use_fp16_decoding)

        self.create_n = kwargs["num_return_sequences"]
        self.summarization = Taskflow("text_summarization", **kwargs)

    def augment(self, sequences):
        """
        Apply augmentation strategy on input sequences.

            Args:
            sequences (str or list(str)):
                Input sequence or list of input sequences.

        """
        if isinstance(sequences, str):
            sequences = [sequences]
        augmented_sequences = self.summarization(sequences)
        return [augmented_sequences[i * self.create_n : (i + 1) * self.create_n] for i in range(len(sequences))]


class SentenceBackTranslate:
    """
    SentenceBackTranslate is a sentence-level data augmentation strategy
    that translates the input sequence into one langugage, and backtranslate
    back into the sourche language by the language models.

    Args:
        src_lang (str):
            The source language of the input sequences.
        tgt_lang (str):
            The target language of the translated sequences.
        max_length (int):
            The max length of the translation.
        batch_size(int):
            The sample number of a mini-batch.
        num_beams (int): The number of beams in the "beam_search"
            strategy. Default to 4.
        use_faster: (bool): Whether to use faster entry of model
            for FasterGeneration. Default to False.
        decode_strategy (str, optional): The decoding strategy in generation.
            Currently, there are three decoding strategies supported:
            "greedy_search", "sampling" and "beam_search". Default to
            "beam_search".
    """

    def __init__(
        self,
        src_lang="zh",
        tgt_lang="en",
        max_length=128,
        batch_size=1,
        num_beams=4,
        use_faster=False,
        decode_strategy="beam_search",
        from_model_name=None,
        to_model_name=None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.use_faster = use_faster
        self.decode_strategy = decode_strategy
        self.from_model_name = from_model_name
        self.to_model_name = to_model_name
        self.MBART_MAP = {
            "ar": "ar_AR",
            "cs": "cs_CZ",
            "de": "de_DE",
            "en": "en_XX",
            "es": "es_XX",
            "et": "et_EE",
            "fi": "fi_FI",
            "fr": "fr_XX",
            "gu": "gu_IN",
            "hi": "hi_IN",
            "it": "it_IT",
            "ja": "ja_XX",
            "kk": "kk_KZ",
            "ko": "ko_KR",
            "lt": "lt_LT",
            "lv": "lv_LV",
            "my": "my_MM",
            "ne": "ne_NP",
            "nl": "nl_XX",
            "ro": "ro_RO",
            "ru": "ru_RU",
            "si": "si_LK",
            "tr": "tr_TR",
            "vi": "vi_VN",
            "zh": "zh_CN",
            "af": "af_ZA",
            "az": "az_AZ",
            "bn": "bn_IN",
            "fa": "fa_IR",
            "he": "he_IL",
            "hr": "hr_HR",
            "id": "id_ID",
            "ka": "ka_GE",
            "km": "km_KH",
            "mk": "mk_MK",
            "ml": "ml_IN",
            "mn": "mn_MN",
            "mr": "mr_IN",
            "pl": "pl_PL",
            "ps": "ps_AF",
            "pt": "pt_XX",
            "sv": "sv_SE",
            "sw": "sw_KE",
            "ta": "ta_IN",
            "te": "te_IN",
            "th": "th_TH",
            "tl": "tl_XX",
            "uk": "uk_UA",
            "ur": "ur_PK",
            "xh": "xh_ZA",
            "gl": "gl_ES",
            "sl": "sl_SI",
        }
        if self.from_model_name is None:
            if tgt_lang == "en":
                self.from_model_name = "mbart-large-50-many-to-one-mmt"
            else:
                self.from_model_name = "mbart-large-50-many-to-many-mmt"

        if to_model_name is None:
            if tgt_lang == "en":
                self.to_model_name = "mbart-large-50-one-to-many-mmt"
            else:
                self.to_model_name = "mbart-large-50-many-to-many-mmt"

        self.from_model = AutoModelForConditionalGeneration.from_pretrained(self.from_model_name)
        self.to_model = AutoModelForConditionalGeneration.from_pretrained(self.to_model_name)
        self.from_tokenizer = AutoTokenizer.from_pretrained(self.from_model_name, src_lang=self.MBART_MAP[src_lang])
        self.to_tokenizer = AutoTokenizer.from_pretrained(self.to_model_name, src_lang=self.MBART_MAP[tgt_lang])
        self.from_model.eval()
        self.to_model.eval()

    def augment(self, sequences):
        """
        Apply augmentation strategy on input sequences.

            Args:
            sequences (str or list(str)):
                Input sequence or list of input sequences.

        """
        if isinstance(sequences, str):
            sequences = [sequences]
        sequences = self._translate(self.from_model, self.from_tokenizer, sequences, self.tgt_lang)
        sequences = self._translate(self.to_model, self.to_tokenizer, sequences, self.src_lang)
        return [[sequence] for sequence in sequences]

    @paddle.no_grad()
    def _translate(self, model, tokenizer, sequences, lang):
        batched_inputs = [sequences[idx : idx + self.batch_size] for idx in range(0, len(sequences), self.batch_size)]
        translated_texts = []
        eos_id = model.mbart.config["eos_token_id"]
        for batched_input in batched_inputs:
            tokenized_input = tokenizer(batched_input, return_tensors="pd", padding=True)["input_ids"]
            outputs = model.generate(
                input_ids=tokenized_input,
                forced_bos_token_id=tokenizer.lang_code_to_id[self.MBART_MAP[lang]],
                decode_strategy=self.decode_strategy,
                num_beams=self.num_beams,
                max_length=self.max_length,
                use_faster=self.use_faster,
            )[0]
            for output in outputs:
                eos = np.where(output.numpy() == eos_id)[0]
                if len(eos) == 0:
                    eos_pos = len(output) - 1
                else:
                    eos_pos = eos[0]
                translated_texts.append(tokenizer.convert_ids_to_string(output[1:eos_pos]))
        return translated_texts


class SentenceBackTranslateAPI:
    """
    SentenceBackTranslateAPI is a sentence-level data augmentation strategy
    that translates the input sequence into one langugage, and backtranslate
    back into the sourche language by baidu translate api.

    Args:
        src_lang (str):
            The source language of the input sequences.
        tgt_lang (str):
            The target language of the translated sequences.
        appid (str):
            Appid for requesting Baidu translation service. (if use your own appid/appkey)
        secretKey (str):
            Secret key for requesting Baidu translation service. (if use your own appid/appkey)
        qps (int):
            Queries per second. (if use your own appid/appkey)
    """

    def __init__(self, src_lang="zh", tgt_lang="en", appid=None, secretKey=None, qps=1):

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.appid = appid
        self.secretKey = secretKey
        self.qps = qps
        self.url = "http://api.fanyi.baidu.com/api/trans/vip/translate"

    def augment(self, sequences):
        """
        Apply augmentation strategy on input sequences.

            Args:
            sequences (str or list(str)):
                Input sequence or list of input sequences.

        """
        if isinstance(sequences, str):
            sequences = [sequences]
        if self.appid is None or self.secretKey is None:
            return self._back_translate_hub(sequences)
        else:
            return self._back_translate_api(sequences)

    def _back_translate_hub(self, sequences):
        try:
            import paddlehub as hub
        except ImportError:
            print(" PaddleHub not installed!")
            import os

            os.system("pip install paddlehub==2.3.1")
            import paddlehub as hub

        module = hub.Module(name="baidu_translate")
        translated_texts = []
        for sequence in sequences:
            sequence = module.translate(sequence, self.src_lang, self.tgt_lang)
            sequence = module.translate(sequence, self.tgt_lang, self.src_lang)
            translated_texts.append([sequence])
        return translated_texts

    def _back_translate_api(self, sequences):

        translated_texts = []
        for sequence in sequences:
            sequence = self._translate_api(sequence, self.src_lang, self.tgt_lang)
            sequence = self._translate_api(sequence, self.tgt_lang, self.src_lang)
            translated_texts.append(sequence)
        return translated_texts

    def _translate_api(self, query, from_lang, to_lang):

        import hashlib
        import random
        import time

        import requests

        # Generate salt and sign
        salt = str(random.randint(32768, 65536))
        sign = self.appid + query + salt + self.secretKey
        sign = hashlib.md5(sign.encode("utf-8")).hexdigest()

        # Build request
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        payload = {
            "appid": f"{self.appid}",
            "q": f"{query}",
            "from": from_lang,
            "to": to_lang,
            "salt": f"{salt}",
            "sign": f"{sign}",
        }

        # Send request
        time.sleep(1 / self.qps)
        try:
            r = requests.post(self.url, params=payload, headers=headers)
            result = r.json()
        except Exception as e:
            error_msg = str(e)
            raise RuntimeError(error_msg)
        if "error_code" in result:
            raise RuntimeError(result)
        return result["trans_result"][0]["dst"]


class SentenceContinue:
    """
    SentenceContinue is a sentence-level data augmentation strategy
    that generates continuation for the input sequence.

    Args:
        model_name (str):
            Model parameter name for summarization task.
        max_length (int):
            The max length of the summarization.
        decode_strategy (str, optional): The decoding strategy in generation.
            Currently, there are three decoding strategies supported:
            "greedy_search", "sampling" and "beam_search". Default to
            "beam_search".
        use_faster: (bool): Whether to use faster entry of model
            for FasterGeneration. Default to False.
        create_n (int):
            Number of augmented sequences.
        batch_size(int):
            The sample number of a mini-batch.
        top_k (int): The number of highest probability tokens to
            keep for top-k-filtering in the "sampling" strategy. Default to
            0, which means no effect.
        top_p (float): The cumulative probability for
            top-p-filtering in the "sampling" strategy. The value should
            satisfy 0 <= top_p < 1. Default to 1.0, which means no
            effect.
        temperature (float): The value used to module the next
            token probabilities in the "sampling" strategy. Default to 1.0,
            which means no effect.
    """

    def __init__(
        self,
        model_name="gpt-cpm-small-cn-distill",
        max_length=64,
        decode_strategy="sampling",
        use_faster=False,
        create_n=1,
        top_k=50,
        temperature=1.0,
        top_p=0.9,
        batch_size=1,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.decode_strategy = decode_strategy
        self.use_faster = use_faster
        self.create_n = create_n
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.convert_ids_to_tokens(self.model.pad_token_id)})

    def augment(self, sequences):
        """
        Apply augmentation strategy on input sequences.

            Args:
            sequences (str or list(str)):
                Input sequence or list of input sequences.

        """
        if isinstance(sequences, str):
            sequences = [sequences]
        return self._generate_continue(sequences, self.model, self.tokenizer)

    @paddle.no_grad()
    def _generate_continue(self, sequences, model, tokenizer):
        batched_inputs = [sequences[idx : idx + self.batch_size] for idx in range(0, len(sequences), self.batch_size)]
        generated_sequences = []
        for batched_input in batched_inputs:
            tokenized_inputs = tokenizer(
                batched_input, return_tensors="pd", padding=True, return_attention_mask=True, return_position_ids=True
            )
            outputs = model.generate(
                **tokenized_inputs,
                max_length=self.max_length,
                decode_strategy=self.decode_strategy,
                use_faster=self.use_faster,
                num_return_sequences=self.create_n,
                top_k=self.top_k,
                temperature=self.temperature,
                top_p=self.top_p,
            )[0]
            for i in range(outputs.shape[0]):
                output = outputs[i].numpy()
                eos = np.where(output == model.eos_token_id)[0]
                if len(eos) == 0:
                    eos_pos = len(output) - 1
                else:
                    eos_pos = eos[0]
                generated_sequences.append(tokenizer.convert_ids_to_string(output[:eos_pos].tolist()))
        augmented_sequences = []
        for i, sequence in enumerate(sequences):
            augmented_sequence = []
            for ii in range(self.create_n):
                continue_sequence = (
                    generated_sequences[i * self.create_n + ii].replace(" ", "").replace("\n", "").replace("\t", "")
                )
                augmented_sequence.append(sequence + continue_sequence)
            augmented_sequences.append(augmented_sequence)
        return augmented_sequences
