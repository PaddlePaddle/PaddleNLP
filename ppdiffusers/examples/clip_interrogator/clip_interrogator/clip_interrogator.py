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

import hashlib
import math
import os
import pickle
from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
import paddle
from paddle.vision import transforms
from PIL import Image
from tqdm import tqdm

from paddlenlp.transformers import CLIPModel, CLIPProcessor

from .blip_decoder import BLIP_Decoder


@dataclass
class Config:
    # blip settings
    blip_pretrained_model_name_or_path: str = "Salesforce/blip-image-captioning-large"
    blip_image_eval_size: int = 384
    blip_max_length: int = 32
    blip_num_beams: int = 8
    blip_min_length: int = 5
    blip_top_p: float = 0.9
    blip_repetition_penalty: float = 1.0
    blip_sample: bool = False

    # clip settings
    clip_pretrained_model_name_or_path: str = "openai/clip-vit-large-patch14"

    # interrogator settings
    cache_path: str = "cache"
    chunk_size: int = 2048
    data_path: str = os.path.join(os.path.dirname(__file__), "data")
    flavor_intermediate_count: int = 2048
    quiet: bool = False  # when quiet progress bars are not shown


class Interrogator:
    def __init__(self, config: Config):
        self.config = config
        # blip model
        self.load_blip_model()
        self.load_clip_model()

    def load_blip_model(self):
        config = self.config
        self.blip_model = BLIP_Decoder(pretrained_model_name_or_path=config.blip_pretrained_model_name_or_path)
        self.blip_model.eval()

    def load_clip_model(self):
        config = self.config

        # clip model
        self.clip_model: CLIPModel = CLIPModel.from_pretrained(config.clip_pretrained_model_name_or_path)
        self.clip_model.eval()
        self.clip_preprocess = CLIPProcessor.from_pretrained(config.clip_pretrained_model_name_or_path)

        sites = [
            "Artstation",
            "behance",
            "cg society",
            "cgsociety",
            "deviantart",
            "dribble",
            "flickr",
            "instagram",
            "pexels",
            "pinterest",
            "pixabay",
            "pixiv",
            "polycount",
            "reddit",
            "shutterstock",
            "tumblr",
            "unsplash",
            "zbrush central",
        ]
        trending_list = [site for site in sites]
        trending_list.extend(["trending on " + site for site in sites])
        trending_list.extend(["featured on " + site for site in sites])
        trending_list.extend([site + " contest winner" for site in sites])

        raw_artists = _load_list(config.data_path, "artists.txt")
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])

        # (TODO, junnyu) we must set pad_token_id to zero
        self.clip_preprocess.tokenizer.pad_token_id = 0
        self.tokenize = partial(
            self.clip_preprocess.tokenizer.__call__,
            return_tensors="pd",
            padding="max_length",
            truncation=True,
            max_length=self.clip_preprocess.tokenizer.model_max_length,
        )
        self.artists = LabelTable(artists, "artists", self.clip_model, self.tokenize, config)
        self.flavors = LabelTable(
            _load_list(config.data_path, "flavors.txt"), "flavors", self.clip_model, self.tokenize, config
        )
        self.mediums = LabelTable(
            _load_list(config.data_path, "mediums.txt"), "mediums", self.clip_model, self.tokenize, config
        )
        self.movements = LabelTable(
            _load_list(config.data_path, "movements.txt"), "movements", self.clip_model, self.tokenize, config
        )
        self.trendings = LabelTable(trending_list, "trendings", self.clip_model, self.tokenize, config)
        self.pad_token_id = self.clip_preprocess.tokenizer.pad_token_id

    def generate_caption(self, pil_image: Image) -> str:
        size = self.config.blip_image_eval_size
        gpu_image = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation="bicubic"),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.clip_preprocess.image_processor.image_mean, self.clip_preprocess.image_processor.image_std
                ),
            ]
        )(pil_image).unsqueeze(0)

        with paddle.no_grad():
            caption = self.blip_model.generate(
                gpu_image,
                sample=self.config.blip_sample,
                num_beams=self.config.blip_num_beams,
                max_length=self.config.blip_max_length,
                min_length=self.config.blip_min_length,
                top_p=self.config.blip_top_p,
                repetition_penalty=self.config.blip_repetition_penalty,
            )
        return caption[0]

    def image_to_features(self, image: Image) -> paddle.Tensor:
        images = self.clip_preprocess(images=image, return_tensors="pd")
        with paddle.no_grad():
            image_features = self.clip_model.get_image_features(images["pixel_values"])
            image_features /= image_features.norm(axis=-1, keepdim=True)
        return image_features

    def interrogate_classic(self, image: Image, max_flavors: int = 3) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)

        medium = self.mediums.rank(image_features, 1)[0]
        artist = self.artists.rank(image_features, 1)[0]
        trending = self.trendings.rank(image_features, 1)[0]
        movement = self.movements.rank(image_features, 1)[0]
        flaves = ", ".join(self.flavors.rank(image_features, max_flavors))

        if caption.startswith(medium):
            prompt = f"{caption} {artist}, {trending}, {movement}, {flaves}"
        else:
            prompt = f"{caption}, {medium} {artist}, {trending}, {movement}, {flaves}"

        return _truncate_to_fit(prompt, self.tokenize, self.pad_token_id)

    def interrogate_fast(self, image: Image, max_flavors: int = 32) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)
        merged = _merge_tables([self.artists, self.flavors, self.mediums, self.movements, self.trendings], self.config)
        tops = merged.rank(image_features, max_flavors)
        return _truncate_to_fit(caption + ", " + ", ".join(tops), self.tokenize, self.pad_token_id)

    def interrogate(self, image: Image, max_flavors: int = 32) -> str:
        caption = self.generate_caption(image)
        image_features = self.image_to_features(image)

        flaves = self.flavors.rank(image_features, self.config.flavor_intermediate_count)
        best_medium = self.mediums.rank(image_features, 1)[0]
        best_artist = self.artists.rank(image_features, 1)[0]
        best_trending = self.trendings.rank(image_features, 1)[0]
        best_movement = self.movements.rank(image_features, 1)[0]

        best_prompt = caption
        best_sim = self.similarity(image_features, best_prompt)

        def check(addition: str) -> bool:
            nonlocal best_prompt, best_sim
            prompt = best_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if sim > best_sim:
                best_sim = sim
                best_prompt = prompt
                return True
            return False

        def check_multi_batch(opts: List[str]):
            nonlocal best_prompt, best_sim
            prompts = []
            for i in range(2 ** len(opts)):
                prompt = best_prompt
                for bit in range(len(opts)):
                    if i & (1 << bit):
                        prompt += ", " + opts[bit]
                prompts.append(prompt)

            t = LabelTable(prompts, None, self.clip_model, self.tokenize, self.config)
            best_prompt = t.rank(image_features, 1)[0]
            best_sim = self.similarity(image_features, best_prompt)

        check_multi_batch([best_medium, best_artist, best_trending, best_movement])

        extended_flavors = set(flaves)
        for i in tqdm(range(max_flavors), desc="Flavor chain", disable=self.config.quiet):
            best = self.rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
            flave = best[len(best_prompt) + 2 :]
            if not check(flave):
                break
            if _prompt_at_max_len(best_prompt, self.tokenize, self.pad_token_id):
                break
            extended_flavors.remove(flave)

        return best_prompt

    def rank_top(self, image_features: paddle.Tensor, text_array: List[str]) -> str:
        text_tokens = self.tokenize(text_array)
        with paddle.no_grad():
            text_features = self.clip_model.get_text_features(text_tokens["input_ids"])
            text_features /= text_features.norm(axis=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return text_array[similarity.argmax().item()]

    def similarity(self, image_features: paddle.Tensor, text: str) -> float:
        text_tokens = self.tokenize([text])
        with paddle.no_grad():
            text_features = self.clip_model.get_text_features(text_tokens["input_ids"])
            text_features /= text_features.norm(axis=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity[0][0].item()


class LabelTable:
    def __init__(self, labels: List[str], desc: str, clip_model, tokenize, config: Config):
        self.chunk_size = config.chunk_size
        self.config = config
        self.embeds = []
        self.labels = labels
        self.tokenize = tokenize

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()

        cache_filepath = None
        if config.cache_path is not None and desc is not None:
            os.makedirs(config.cache_path, exist_ok=True)
            sanitized_name = config.clip_pretrained_model_name_or_path.replace("/", "_").replace("@", "_")
            cache_filepath = os.path.join(config.cache_path, f"{sanitized_name}_{desc}.pkl")
            if desc is not None and os.path.exists(cache_filepath):
                with open(cache_filepath, "rb") as f:
                    try:
                        data = pickle.load(f)
                        if data.get("hash") == hash:
                            self.labels = data["labels"]
                            self.embeds = data["embeds"]
                    except Exception as e:
                        print(f"Error loading cached table {desc}: {e}")

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels) / config.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.config.quiet):
                text_tokens = self.tokenize(chunk.tolist())
                with paddle.no_grad():
                    text_features = clip_model.get_text_features(text_tokens["input_ids"])
                    text_features /= text_features.norm(axis=-1, keepdim=True)
                    text_features = text_features.cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if cache_filepath is not None:
                with open(cache_filepath, "wb") as f:
                    pickle.dump(
                        {
                            "labels": self.labels,
                            "embeds": self.embeds,
                            "hash": hash,
                            "model": config.clip_pretrained_model_name_or_path,
                        },
                        f,
                    )

    def _rank(self, image_features: paddle.Tensor, text_embeds: paddle.Tensor, top_count: int = 1) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = paddle.to_tensor(text_embeds)
        similarity = image_features @ text_embeds.T
        _, top_labels = similarity.cast("float32").topk(top_count, axis=-1)
        top_labels = top_labels.tolist()
        return [top_labels[0][i] for i in range(top_count)]

    def rank(self, image_features: paddle.Tensor, top_count: int = 1) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels) / self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.config.quiet):
            start = chunk_idx * self.chunk_size
            stop = min(start + self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start + i] for i in tops])
            top_embeds.extend([self.embeds[start + i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]


def _load_list(data_path: str, filename: str) -> List[str]:
    with open(os.path.join(data_path, filename), "r", encoding="utf-8", errors="replace") as f:
        items = [line.strip() for line in f.readlines()]
    return items


def _merge_tables(tables: List[LabelTable], config: Config) -> LabelTable:
    m = LabelTable([], None, None, None, config)
    for table in tables:
        m.labels.extend(table.labels)
        m.embeds.extend(table.embeds)
    return m


def _prompt_at_max_len(text: str, tokenize, pad_token_id: int = 0) -> bool:
    tokens = tokenize([text])["input_ids"]
    return tokens[0][-1] != pad_token_id


def _truncate_to_fit(text: str, tokenize, pad_token_id) -> str:
    parts = text.split(", ")
    new_text = parts[0]
    for part in parts[1:]:
        if _prompt_at_max_len(new_text + part, tokenize, pad_token_id):
            break
        new_text += ", " + part
    return new_text
