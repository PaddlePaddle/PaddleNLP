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

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import paddle
from PIL import Image
from pipelines.schema import Document
from tqdm.auto import tqdm

from paddlenlp import Taskflow

logger = logging.getLogger(__name__)
FilterType = Dict[str, Union[Dict[str, Any], List[Any], str, int, float, bool]]


DOCUMENT_CONVERTERS = {
    # NOTE: Keep this '?' cleaning step, it needs to be double-checked for impact on the inference results.
    "text": lambda doc: doc.content[:-1] if doc.content[-1] == "?" else doc.content,
    "image": lambda doc: Image.open(doc.content),
}

CAN_EMBED_META = ["text"]


class MultiModalEmbedder:
    def __init__(
        self,
        embedding_models: Dict[str, Union[Path, str]],  # replace str with ContentTypes starting from Python3.8
        feature_extractors_params: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: int = 16,
        embed_meta_fields: List[str] = ["name"],
        progress_bar: bool = True,
    ):
        """
        Init the Retriever and all its models from a local or remote model checkpoint.
        :param embedding_models: A dictionary matching a local path or remote name of encoder checkpoint with
            the content type it should handle ("text",  "image", etc...).
            Expected input format: `{'text': 'name_or_path_to_text_model', 'image': 'name_or_path_to_image_model', ... }`
            Keep in mind that the models should output in the same embedding space for this retriever to work.
        :param feature_extractors_params: A dictionary matching a content type ("text",  "image" and so on) with the
            parameters of its own feature extractor if the model requires one.
            Expected input format: `{'text': {'param_name': 'param_value', ...}, 'image': {'param_name': 'param_value', ...}, ...}`
        :param batch_size: Number of questions or passages to encode at once. In case of multiple GPUs, this will be the total batch size.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / image to a text pair that is
                                  then used to create the embedding.
                                  This is the approach used in the original paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        """
        super().__init__()

        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.embed_meta_fields = embed_meta_fields

        feature_extractors_params = {
            content_type: {"max_length": 256, **(feature_extractors_params or {}).get(content_type, {})}
            for content_type in ["text", "image"]  # FIXME get_args(ContentTypes) from Python3.8 on
        }

        self.models = {}  # replace str with ContentTypes starting from Python3.8
        for content_type, embedding_model in embedding_models.items():
            if content_type in ["text", "image"]:
                self.models[content_type] = Taskflow("feature_extraction", model=embedding_model)
            else:
                raise ValueError(f"{content_type} is not a supported content.")

        # Check embedding sizes for models: they must all match
        if len(self.models) > 1:
            sizes = {model.embedding_dim for model in self.models.values()}
            if None in sizes:
                logger.warning(
                    "Pipelines could not find the output embedding dimensions for '%s'. "
                    "Dimensions won't be checked before computing the embeddings.",
                    ", ".join(
                        {
                            str(model.model_name_or_path)
                            for model in self.models.values()
                            if model.embedding_dim is None
                        }
                    ),
                )
            elif len(sizes) > 1:
                embedding_sizes: Dict[int, List[str]] = {}
                for model in self.models.values():
                    embedding_sizes[model.embedding_dim] = embedding_sizes.get(model.embedding_dim, []) + [
                        str(model.model_name_or_path)
                    ]
                raise ValueError(f"Not all models have the same embedding size: {embedding_sizes}")

    def embed(self, documents: List[Document], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Create embeddings for a list of documents using the relevant encoder for their content type.
        :param documents: Documents to embed.
        :return: Embeddings, one per document, in the form of a np.array
        """
        batch_size = batch_size if batch_size is not None else self.batch_size

        all_embeddings = []
        for batch_index in tqdm(
            iterable=range(0, len(documents), batch_size),
            unit=" Docs",
            desc="Create embeddings",
            position=1,
            leave=False,
            disable=not self.progress_bar,
        ):
            docs_batch = documents[batch_index : batch_index + batch_size]
            data_by_type = self._docs_to_data(documents=docs_batch)

            # Get output for each model
            outputs_by_type: Dict[str, paddle.Tensor] = {}  # replace str with ContentTypes starting Python3.8
            for data_type, data in data_by_type.items():

                model = self.models.get(data_type)
                if not model:
                    raise Exception(
                        f"Some data of type {data_type} was passed, but no model capable of handling such data was "
                        f"initialized. Initialized models: {', '.join(self.models.keys())}"
                    )
                outputs_by_type[data_type] = model(data)["features"]
            # Check the output sizes
            embedding_sizes = [output.shape[-1] for output in outputs_by_type.values()]

            if not all(embedding_size == embedding_sizes[0] for embedding_size in embedding_sizes):
                raise Exception(
                    "Some of the models are using a different embedding size. They should all match. "
                    f"Embedding sizes by model: "
                    f"{ {name: output.shape[-1] for name, output in outputs_by_type.items()} }"
                )

            # Combine the outputs in a single matrix
            outputs = paddle.stack(list(outputs_by_type.values()))
            embeddings = outputs.reshape([-1, embedding_sizes[0]])
            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings)

    def _docs_to_data(
        self, documents: List[Document]
    ) -> Dict[str, List[Any]]:  # FIXME replace str to ContentTypes from Python3.8
        """
        Extract the data to embed from each document and return them classified by content type.
        :param documents: The documents to prepare fur multimodal embedding.
        :return: A dictionary containing one key for each content type, and a list of data extracted
            from each document, ready to be passed to the feature extractor (for example the content
            of a text document, a linearized table, a PIL image object, and so on)
        """
        docs_data: Dict[str, List[Any]] = {  # FIXME replace str to ContentTypes from Python3.8
            key: [] for key in ["text", "image"]
        }  # FIXME get_args(ContentTypes) from Python3.8 on
        for doc in documents:
            try:
                document_converter = DOCUMENT_CONVERTERS[doc.content_type]
            except KeyError:
                raise Exception(
                    f"Unknown content type '{doc.content_type}'. Known types: 'text', 'image'."  # FIXME {', '.join(get_args(ContentTypes))}"  from Python3.8 on
                )

            data = document_converter(doc)

            if doc.content_type in CAN_EMBED_META:
                meta = [v for k, v in (doc.meta or {}).items() if k in self.embed_meta_fields]
                data = f"{' '.join(meta)} {data}" if meta else data

            docs_data[doc.content_type].append(data)

        return {key: values for key, values in docs_data.items() if values}
