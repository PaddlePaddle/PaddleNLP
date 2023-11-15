# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
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

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


# if typing.TYPE_CHECKING:
import ast
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import mmh3
import numpy as np
import pandas as pd

#     from dataclasses import dataclass  # type: ignore
# else:
#     # We are using Pydantic dataclasses instead of vanilla Python's
#     # See #1598 for the reasons behind this choice & performance considerations
#     from pydantic.dataclasses import dataclass
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder

logger = logging.getLogger(__name__)


# BaseConfig.arbitrary_types_allowed = True
class Config:
    arbitrary_types_allowed = True


#: Types of content_types supported
ContentTypes = Literal["text", "image"]
FilterType = Dict[str, Union[Dict[str, Any], List[Any], str, int, float, bool]]


@dataclass(config=Config)
class Document:
    content: Union[str, pd.DataFrame]
    content_type: Literal["text", "table", "image"]
    id: str
    meta: Dict[str, Any]
    score: Optional[float] = None
    embedding: Optional[np.ndarray] = None
    id_hash_keys: Optional[List[str]] = None

    # We use a custom init here as we want some custom logic. The annotations above are however still needed in order
    # to use some dataclass magic like "asdict()". See https://www.python.org/dev/peps/pep-0557/#custom-init-method
    # They also help in annotating which object attributes will always be present (e.g. "id") even though they
    # don't need to passed by the user in init and are rather initialized automatically in the init
    def __init__(
        self,
        content: Union[str, pd.DataFrame],
        content_type: Literal["text", "table", "image"] = "text",
        id: Optional[str] = None,
        score: Optional[float] = None,
        meta: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None,
        id_hash_keys: Optional[List[str]] = None,
    ):
        """
        One of the core data classes in PIPELINES. It's used to represent documents / passages in a standardized way within PIPELINES.
        Documents are stored in DocumentStores, are returned by Retrievers, are the input for Readers and are used in
        many other places that manipulate or interact with document-level data.

        Note: There can be multiple Documents originating from one file (e.g. PDF), if you split the text
        into smaller passages. We'll have one Document per passage in this case.

        Each document has a unique ID. This can be supplied by the user or generated automatically.
        It's particularly helpful for handling of duplicates and referencing documents in other objects (e.g. Labels)

        There's an easy option to convert from/to dicts via `from_dict()` and `to_dict`.

        :param content: Content of the document. For most cases, this will be text, but it can be a table or image.
        :param content_type: One of "image", "table" or "image". PIPELINES components can use this to adjust their
                             handling of Documents and check compatibility.
        :param id: Unique ID for the document. If not supplied by the user, we'll generate one automatically by
                   creating a hash from the supplied text. This behaviour can be further adjusted by `id_hash_keys`.
        :param score: The relevance score of the Document determined by a model (e.g. Retriever or Re-Ranker).
                      In the range of [0,1], where 1 means extremely relevant.
        :param meta: Meta fields for a document like name, url, or author in the form of a custom dict (any keys and values allowed).
        :param embedding: Vector encoding of the text
        :param id_hash_keys: Generate the document id from a custom list of strings that refere to the documents attributes.
                             If you want ensure you don't have duplicate documents in your DocumentStore but texts are
                             not unique, you can modify the metadata and pass e.g. "meta" to this field (e.g. ["content", "meta"]).
                             In this case the id will be generated by using the content and the defined metadata.
        """

        if content is None:
            raise ValueError("Can't create 'Document': Mandatory 'content' field is None")

        self.content = content
        self.content_type = content_type
        self.score = score
        self.meta = meta or {}

        allowed_hash_key_attributes = ["content", "content_type", "score", "meta", "embedding"]

        if id_hash_keys is not None:
            if not set(id_hash_keys) <= set(allowed_hash_key_attributes):  # type: ignore
                raise ValueError(
                    f"You passed custom strings {id_hash_keys} to id_hash_keys which is deprecated. Supply instead a list of Document's attribute names that the id should be based on (e.g. {allowed_hash_key_attributes})."
                )

        if embedding is not None:
            embedding = np.asarray(embedding)
        self.embedding = embedding

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id: str = str(id)
        else:
            self.id: str = self._get_id(id_hash_keys=id_hash_keys)

    def _get_id(self, id_hash_keys: Optional[List[str]] = None):
        """
        Generate the id of a document by creating the hash of strings. By default the content of a document is
        used to generate the hash. There are two ways of modifying the generated id of a document. Either static keys
        or a selection of the content.
        :param id_hash_keys: Optional list of fields that should be dynamically used to generate the hash.
        """

        if id_hash_keys is None:
            return "{:02x}".format(mmh3.hash128(str(self.content), signed=False))

        final_hash_key = ""
        for attr in id_hash_keys:
            final_hash_key += ":" + str(getattr(self, attr))

        if final_hash_key == "":
            raise ValueError(
                "Cant't create 'Document': 'id_hash_keys' must contain at least one of ['content', 'meta']"
            )

        return "{:02x}".format(mmh3.hash128(final_hash_key, signed=False))

    def to_dict(self, field_map={}) -> Dict:
        """
        Convert Document to dict. An optional field_map can be supplied to change the names of the keys in the
        resulting dict. This way you can work with standardized Document objects in PIPELINES, but adjust the format that
        they are serialized / stored in other places (e.g. elasticsearch)
        Example:
        | doc = Document(content="some text", content_type="text")
        | doc.to_dict(field_map={"custom_content_field": "content"})
        | >>> {"custom_content_field": "some text", content_type": "text"}

        :param field_map: Dict with keys being the custom target keys and values being the standard Document attributes
        :return: dict with content of the Document
        """
        inv_field_map = {v: k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            if k == "content":
                # Convert pd.DataFrame to list of rows for serialization
                if self.content_type == "table" and isinstance(self.content, pd.DataFrame):
                    v = [self.content.columns.tolist()] + self.content.values.tolist()
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, dict, field_map={}, id_hash_keys=None):
        """
        Create Document from dict. An optional field_map can be supplied to adjust for custom names of the keys in the
        input dict. This way you can work with standardized Document objects in PIPELINES, but adjust the format that
        they are serialized / stored in other places (e.g. elasticsearch)
        Example:
        | my_dict = {"custom_content_field": "some text", content_type": "text"}
        | Document.from_dict(my_dict, field_map={"custom_content_field": "content"})

        :param field_map: Dict with keys being the custom target keys and values being the standard Document attributes
        :return: dict with content of the Document
        """

        _doc = dict.copy()
        init_args = ["content", "content_type", "id", "score", "question", "meta", "embedding"]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        # copy additional fields into "meta"
        for k, v in _doc.items():
            if k not in init_args and k not in field_map:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _new_doc = {}
        for k, v in _doc.items():
            if k in init_args:
                _new_doc[k] = v
            elif k in field_map:
                k = field_map[k]
                _new_doc[k] = v

        if _doc.get("id") is None:
            _new_doc["id_hash_keys"] = id_hash_keys

        # Convert list of rows to pd.DataFrame
        if _new_doc.get("content_type", None) == "table" and isinstance(_new_doc["content"], list):
            _new_doc["content"] = pd.DataFrame(columns=_new_doc["content"][0], data=_new_doc["content"][1:])

        return cls(**_new_doc)

    def to_json(self, field_map={}) -> str:
        d = self.to_dict(field_map=field_map)
        j = json.dumps(d, cls=NumpyEncoder)
        return j

    @classmethod
    def from_json(cls, data: str, field_map={}):
        d = json.loads(data)
        return cls.from_dict(d, field_map=field_map)

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, "content", None) == self.content
            and getattr(other, "content_type", None) == self.content_type
            and getattr(other, "id", None) == self.id
            and getattr(other, "score", None) == self.score
            and getattr(other, "meta", None) == self.meta
            and np.array_equal(getattr(other, "embedding", None), self.embedding)
            and getattr(other, "id_hash_keys", None) == self.id_hash_keys
        )

    def __repr__(self):
        return f"<Document: {str(self.to_dict())}>"

    def __str__(self):
        # In some cases, self.content is None (therefore not subscriptable)
        if not self.content:
            return f"<Document: id={self.id}, content=None>"
        return f"<Document: id={self.id}, content='{self.content[:100]} {'...' if len(self.content) > 100 else ''}'>"

    def __lt__(self, other):
        """Enable sorting of Documents by score"""
        return self.score < other.score


@dataclass
class Span:
    start: int
    end: int
    """
    Defining a sequence of characters (Text span) or cells (Table span) via start and end index.
    For extractive QA: Character where answer starts/ends
    For TableQA: Cell where the answer starts/ends (counted from top left to bottom right of table)

    :param start: Position where the span starts
    :param end:  Position where the spand ends
    """


@dataclass(config=Config)
class Answer:
    answer: str
    type: Literal["generative", "extractive", "other"] = "extractive"
    score: Optional[float] = None
    context: Optional[Union[str, pd.DataFrame]] = None
    offsets_in_document: Optional[List[Span]] = None
    offsets_in_context: Optional[List[Span]] = None
    document_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    """
    The fundamental object in PIPELINES to represent any type of Answers (e.g. extractive QA, generative QA or TableQA).
    For example, it's used within some Nodes like the Reader, but also in the REST API.

    :param answer: The answer string. If there's no possible answer (aka "no_answer" or "is_impossible) this will be an empty string.
    :param type: One of ("generative", "extractive", "other"): Whether this answer comes from an extractive model
                 (i.e. we can locate an exact answer string in one of the documents) or from a generative model
                 (i.e. no pointer to a specific document, no offsets ...).
    :param score: The relevance score of the Answer determined by a model (e.g. Reader or Generator).
                  In the range of [0,1], where 1 means extremely relevant.
    :param context: The related content that was used to create the answer (i.e. a text passage, part of a table, image ...)
    :param offsets_in_document: List of `Span` objects with start and end positions of the answer **in the
                                document** (as stored in the document store).
                                For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start
                                For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
                                (Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here)
    :param offsets_in_context: List of `Span` objects with start and end positions of the answer **in the
                                context** (i.e. the surrounding text/table of a certain window size).
                                For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start
                                For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
                                (Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here)
    :param document_id: ID of the document that the answer was located it (if any)
    :param meta: Dict that can be used to associate any kind of custom meta data with the answer.
                 In extractive QA, this will carry the meta data of the document where the answer was found.
    """

    def __post_init__(self):
        # In case offsets are passed as dicts rather than Span objects we convert them here
        # For example, this is used when instantiating an object via from_json()
        if self.offsets_in_document is not None:
            self.offsets_in_document = [Span(**e) if isinstance(e, dict) else e for e in self.offsets_in_document]
        if self.offsets_in_context is not None:
            self.offsets_in_context = [Span(**e) if isinstance(e, dict) else e for e in self.offsets_in_context]

        if self.meta is None:
            self.meta = {}

    def __lt__(self, other):
        """Enable sorting of Answers by score"""
        return self.score < other.score

    def __str__(self):
        # self.context might be None (therefore not subscriptable)
        if not self.context:
            return f"<Answer: answer='{self.answer}', score={self.score}, context=None>"
        return f"<Answer: answer='{self.answer}', score={self.score}, context='{self.context[:50]}{'...' if len(self.context) > 50 else ''}'>"

    def __repr__(self):
        return f"<Answer {asdict(self)}>"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict: dict):
        return _pydantic_dataclass_from_dict(dict=dict, pydantic_dataclass_type=cls)

    def to_json(self):
        return json.dumps(self, default=pydantic_encoder)

    @classmethod
    def from_json(cls, data):
        if type(data) == str:
            data = json.loads(data)
        return cls.from_dict(data)


@dataclass
class Label:
    id: str
    query: str
    document: Document
    is_correct_answer: bool
    is_correct_document: bool
    origin: Literal["user-feedback", "gold-label"]
    answer: Optional[Answer] = None
    no_answer: Optional[bool] = None
    pipeline_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    meta: Optional[dict] = None
    filters: Optional[dict] = None

    # We use a custom init here as we want some custom logic. The annotations above are however still needed in order
    # to use some dataclass magic like "asdict()". See https://www.python.org/dev/peps/pep-0557/#custom-init-method
    def __init__(
        self,
        query: str,
        document: Document,
        is_correct_answer: bool,
        is_correct_document: bool,
        origin: Literal["user-feedback", "gold-label"],
        answer: Optional[Answer],
        id: Optional[str] = None,
        no_answer: Optional[bool] = None,
        pipeline_id: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        meta: Optional[dict] = None,
        filters: Optional[dict] = None,
    ):
        """
        Object used to represent label/feedback in a standardized way within PIPELINES.
        This includes labels from dataset like SQuAD, annotations from labeling tools,
        or, user-feedback from the PIPELINES REST API.

        :param query: the question (or query) for finding answers.
        :param document:
        :param answer: the answer object.
        :param is_correct_answer: whether the sample is positive or negative.
        :param is_correct_document: in case of negative sample(is_correct_answer is False), there could be two cases;
                                    incorrect answer but correct document & incorrect document. This flag denotes if
                                    the returned document was correct.
        :param origin: the source for the labels. It can be used to later for filtering.
        :param id: Unique ID used within the DocumentStore. If not supplied, a uuid will be generated automatically.
        :param no_answer: whether the question in unanswerable.
        :param pipeline_id: pipeline identifier (any str) that was involved for generating this label (in-case of user feedback).
        :param created_at: Timestamp of creation with format yyyy-MM-dd HH:mm:ss.
                           Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S").
        :param created_at: Timestamp of update with format yyyy-MM-dd HH:mm:ss.
                           Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S")
        :param meta: Meta fields like "annotator_name" in the form of a custom dict (any keys and values allowed).
        :param filters: filters that should be applied to the query to rule out non-relevant documents. For example, if there are different correct answers
                        in a DocumentStore depending on the retrieved document and the answer in this label is correct only on condition of the filters.
        """

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())

        if created_at is None:
            created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.created_at = created_at

        self.updated_at = updated_at
        self.query = query
        self.answer = answer
        self.document = document
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin

        # Remove
        # self.document_id = document_id
        # self.offset_start_in_doc = offset_start_in_doc

        # If an Answer is provided we need to make sure that it's consistent with the `no_answer` value
        # TODO: reassess if we want to enforce Span.start=0 and Span.end=0 for no_answer=True
        if self.answer is not None:
            if no_answer is True:
                if self.answer.answer != "" or self.answer.context:
                    raise ValueError(
                        f"Got no_answer == True while there seems to be an possible Answer: {self.answer}"
                    )
            elif no_answer is False:
                if self.answer.answer == "":
                    raise ValueError(
                        f"Got no_answer == False while there seems to be no possible Answer: {self.answer}"
                    )
            else:
                # Automatically infer no_answer from Answer object
                no_answer = self.answer.answer == "" or self.answer.answer is None

        self.no_answer = no_answer

        # TODO autofill answer.document_id if Document is provided

        self.pipeline_id = pipeline_id
        if not meta:
            self.meta = {}
        else:
            self.meta = meta
        self.filters = filters

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict: dict):
        return _pydantic_dataclass_from_dict(dict=dict, pydantic_dataclass_type=cls)

    def to_json(self):
        return json.dumps(self, default=pydantic_encoder)

    @classmethod
    def from_json(cls, data):
        if type(data) == str:
            data = json.loads(data)
        return cls.from_dict(data)

    # define __eq__ and __hash__ functions to deduplicate Label Objects
    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, "query", None) == self.query
            and getattr(other, "answer", None) == self.answer
            and getattr(other, "is_correct_answer", None) == self.is_correct_answer
            and getattr(other, "is_correct_document", None) == self.is_correct_document
            and getattr(other, "origin", None) == self.origin
            and getattr(other, "document", None) == self.document
            and getattr(other, "no_answer", None) == self.no_answer
            and getattr(other, "pipeline_id", None) == self.pipeline_id
        )

    def __hash__(self):
        return hash(
            self.query
            + str(self.answer)
            + str(self.is_correct_answer)
            + str(self.is_correct_document)
            + str(self.origin)
            + str(self.document)
            + str(self.no_answer)
            + str(self.pipeline_id)
        )

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return f"<Label: {self.to_dict()}>"


@dataclass
class MultiLabel:
    labels: List[Label]
    query: str
    answers: List[str]
    no_answer: bool
    document_ids: List[str]
    document_contents: List[str]
    gold_offsets_in_contexts: List[Dict]
    gold_offsets_in_documents: List[Dict]

    def __init__(self, labels: List[Label], drop_negative_labels=False, drop_no_answers=False):
        """
        There are often multiple `Labels` associated with a single query. For example, there can be multiple annotated
        answers for one question or multiple documents contain the information you want for a query.
        This class is "syntactic sugar" that simplifies the work with such a list of related Labels.
        It stored the original labels in MultiLabel.labels and provides additional aggregated attributes that are
        automatically created at init time. For example, MultiLabel.no_answer allows you to easily access if any of the
        underlying Labels provided a text answer and therefore demonstrates that there is indeed a possible answer.

        :param labels: A list of labels that belong to a similar query and shall be "grouped" together
        :param drop_negative_labels: Whether to drop negative labels from that group (e.g. thumbs down feedback from UI)
        :param drop_no_answers: Whether to drop labels that specify the answer is impossible
        """
        # drop duplicate labels and remove negative labels if needed.
        labels = list(set(labels))
        if drop_negative_labels:
            is_positive_label = lambda l: (l.is_correct_answer and l.is_correct_document) or (  # noqa: E731
                l.answer is None and l.is_correct_document
            )
            labels = [l for l in labels if is_positive_label(l)]

        if drop_no_answers:
            labels = [l for l in labels if l.no_answer is False]

        self.labels = labels

        self.query = self._aggregate_labels(key="query", must_be_single_value=True)[0]
        self.filters = self._aggregate_labels(key="filters", must_be_single_value=True)[0]
        self.id = hash((self.query, json.dumps(self.filters, sort_keys=True).encode()))

        # Currently no_answer is only true if all labels are "no_answers", we could later introduce a param here to let
        # users decided which aggregation logic they want
        self.no_answer = False not in [l.no_answer for l in self.labels]

        # Answer strings and offsets cleaned for no_answers:
        # If there are only no_answers, offsets are empty and answers will be a single empty string
        # which equals the no_answers representation of reader nodes.
        if self.no_answer:
            self.answers = [""]
            self.gold_offsets_in_documents: List[dict] = []
            self.gold_offsets_in_contexts: List[dict] = []
        else:
            answered = [l.answer for l in self.labels if not l.no_answer and l.answer is not None]
            self.answers = [answer.answer for answer in answered]
            self.gold_offsets_in_documents = []
            self.gold_offsets_in_contexts = []
            for answer in answered:
                if answer.offsets_in_document is not None:
                    for span in answer.offsets_in_document:
                        self.gold_offsets_in_documents.append({"start": span.start, "end": span.end})
                if answer.offsets_in_context is not None:
                    for span in answer.offsets_in_context:
                        self.gold_offsets_in_contexts.append({"start": span.start, "end": span.end})

        # There are two options here to represent document_ids:
        # taking the id from the document of each label or taking the document_id of each label's answer.
        # We take the former as labels without answers are allowed.
        #
        # For no_answer cases document_store.add_eval_data() currently adds all documents coming from the SQuAD paragraph's context
        # as separate no_answer labels, and thus with document.id but without answer.document_id.
        # If we do not exclude them from document_ids this would be problematic for retriever evaluation as they do not contain the answer.
        # Hence, we exclude them here as well.
        self.document_ids = [l.document.id for l in self.labels if not l.no_answer]
        self.document_contents = [l.document.content for l in self.labels if not l.no_answer]

    def _aggregate_labels(self, key, must_be_single_value=True) -> List[Any]:
        if any(isinstance(getattr(l, key), dict) for l in self.labels):
            # dict is not hashable so we collect unique filters via looping through all labels
            unique_values = []
            for l in self.labels:
                if l.filters not in unique_values:
                    unique_values.append(l.filters)
        else:
            unique_values = list({getattr(l, key) for l in self.labels})
        if must_be_single_value and len(unique_values) > 1:
            raise ValueError(
                f"Tried to combine attribute '{key}' of Labels, but found multiple different values: {unique_values}"
            )
        return unique_values

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict: dict):
        return _pydantic_dataclass_from_dict(dict=dict, pydantic_dataclass_type=cls)

    def to_json(self):
        return json.dumps(self, default=pydantic_encoder)

    @classmethod
    def from_json(cls, data):
        if type(data) == str:
            data = json.loads(data)
        return cls.from_dict(data)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return f"<MultiLabel: {self.to_dict()}>"


def _pydantic_dataclass_from_dict(dict: dict, pydantic_dataclass_type) -> Any:
    """
    Constructs a pydantic dataclass from a dict incl. other nested dataclasses.
    This allows simple de-serialization of pydentic dataclasses from json.
    :param dict: Dict containing all attributes and values for the dataclass.
    :param pydantic_dataclass_type: The class of the dataclass that should be constructed (e.g. Document)
    """
    base_model = pydantic_dataclass_type.__pydantic_model__.parse_obj(dict)
    base_mode_fields = base_model.__fields__

    values = {}
    for base_model_field_name, base_model_field in base_mode_fields.items():
        value = getattr(base_model, base_model_field_name)
        values[base_model_field_name] = value

    dataclass_object = pydantic_dataclass_type(**values)
    return dataclass_object


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class EvaluationResult:
    def __init__(self, node_results: Dict[str, pd.DataFrame] = None) -> None:
        """
        Convenience class to store, pass and interact with results of a pipeline evaluation run (e.g. pipeline.eval()).
        Detailed results are stored as one dataframe per node. This class makes them more accessible and provides
        convenience methods to work with them.
        For example, you can calculate eval metrics, get detailed reports or simulate different top_k settings.

        Example:
        ```python
        | eval_results = pipeline.eval(...)
        |
        | # derive detailed metrics
        | eval_results.calculate_metrics()
        |
        | # show summary of incorrect queries
        | eval_results.wrong_examples()
        ```

        Each row of the underlying DataFrames contains either an answer or a document that has been retrieved during evaluation.
        Rows are enriched with basic infos like rank, query, type or node.
        Additional answer or document specific evaluation infos like gold labels
        and metrics depicting whether the row matches the gold labels are included, too.
        The DataFrames have the following schema:
        - multilabel_id: the id of the multilabel, which is unique for the pair of query and filters
        - query: the query
        - filters: the filters used with the query
        - gold_answers (answers only): the answers to be given
        - answer (answers only): the answer
        - context (answers only): the surrounding context of the answer within the document
        - exact_match (answers only): metric depicting if the answer exactly matches the gold label
        - f1 (answers only): metric depicting how well the answer overlaps with the gold label on token basis
        - sas (answers only, optional): metric depciting how well the answer matches the gold label on a semantic basis
        - gold_document_contents (documents only): the contents of the gold documents
        - content (documents only): the content of the document
        - gold_id_match (documents only): metric depicting whether one of the gold document ids matches the document
        - answer_match (documents only): metric depicting whether the document contains the answer
        - gold_id_or_answer_match (documents only): metric depicting whether one of the former two conditions are met
        - rank: rank or 1-based-position in result list
        - document_id: the id of the document that has been retrieved or that contained the answer
        - gold_document_ids: the documents to be retrieved
        - offsets_in_document (answers only): the position or offsets within the document the answer was found
        - gold_offsets_in_documents (answers only): the positon or offsets of the gold answer within the document
        - type: 'answer' or 'document'
        - node: the node name
        - eval_mode: evaluation mode depicting whether the evaluation was executed in integrated or isolated mode.
                     Check pipeline.eval()'s add_isolated_node_eval param for more information.

        :param node_results: the evaluation Dataframes per pipeline node
        """
        self.node_results: Dict[str, pd.DataFrame] = {} if node_results is None else node_results

    def __getitem__(self, key: str):
        return self.node_results.__getitem__(key)

    def __delitem__(self, key: str):
        self.node_results.__delitem__(key)

    def __setitem__(self, key: str, value: pd.DataFrame):
        self.node_results.__setitem__(key, value)

    def __contains__(self, key: str):
        return self.node_results.keys().__contains__(key)

    def __len__(self):
        return self.node_results.__len__()

    def append(self, key: str, value: pd.DataFrame):
        if value is not None and len(value) > 0:
            if key in self.node_results:
                self.node_results[key] = pd.concat([self.node_results[key], value])
            else:
                self.node_results[key] = value

    def calculate_metrics(
        self,
        simulated_top_k_reader: int = -1,
        simulated_top_k_retriever: int = -1,
        doc_relevance_col: str = "gold_id_match",
        eval_mode: str = "integrated",
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates proper metrics for each node.

        For document returning nodes default metrics are:
        - mrr (Mean Reciprocal Rank: see https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
        - map (Mean Average Precision: see https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Mean_average_precision)
        - ndcg (Normalized Discounted Cumulative Gain: see https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
        - precision (Precision: How many of the returned documents were relevant?)
        - recall_multi_hit (Recall according to Information Retrieval definition: How many of the relevant documents were retrieved per query?)
        - recall_single_hit (Recall for Question Answering: How many of the queries returned at least one relevant document?)

        For answer returning nodes default metrics are:
        - exact_match (How many of the queries returned the exact answer?)
        - f1 (How well do the returned results overlap with any gold answer on token basis?)
        - sas if a SAS model has bin provided during during pipeline.eval() (How semantically similar is the prediction to the gold answers?)

        Lower top_k values for reader and retriever than the actual values during the eval run can be simulated.
        E.g. top_1_f1 for reader nodes can be calculated by setting simulated_top_k_reader=1.

        Results for reader nodes with applied simulated_top_k_retriever should be considered with caution
        as there are situations the result can heavily differ from an actual eval run with corresponding top_k_retriever.

        :param simulated_top_k_reader: simulates top_k param of reader
        :param simulated_top_k_retriever: simulates top_k param of retriever.
            remarks: there might be a discrepancy between simulated reader metrics and an actual pipeline run with retriever top_k
        :param doc_relevance_col: column in the underlying eval table that contains the relevance criteria for documents.
            values can be: 'gold_id_match', 'answer_match', 'gold_id_or_answer_match'
        :param eval_mode: the input on which the node was evaluated on.
            Usually nodes get evaluated on the prediction provided by its predecessor nodes in the pipeline (value='integrated').
            However, as the quality of the node itself can heavily depend on the node's input and thus the predecessor's quality,
            you might want to simulate a perfect predecessor in order to get an independent upper bound of the quality of your node.
            For example when evaluating the reader use value='isolated' to simulate a perfect retriever in an ExtractiveQAPipeline.
            Values can be 'integrated', 'isolated'.
            Default value is 'integrated'.
        """
        return {
            node: self._calculate_node_metrics(
                df,
                simulated_top_k_reader=simulated_top_k_reader,
                simulated_top_k_retriever=simulated_top_k_retriever,
                doc_relevance_col=doc_relevance_col,
                eval_mode=eval_mode,
            )
            for node, df in self.node_results.items()
        }

    def wrong_examples(
        self,
        node: str,
        n: int = 3,
        simulated_top_k_reader: int = -1,
        simulated_top_k_retriever: int = -1,
        doc_relevance_col: str = "gold_id_match",
        document_metric: str = "recall_single_hit",
        answer_metric: str = "f1",
        eval_mode: str = "integrated",
    ) -> List[Dict]:
        """
        Returns the worst performing queries.
        Worst performing queries are calculated based on the metric
        that is either a document metric or an answer metric according to the node type.

        Lower top_k values for reader and retriever than the actual values during the eval run can be simulated.
        See calculate_metrics() for more information.

        :param simulated_top_k_reader: simulates top_k param of reader
        :param simulated_top_k_retriever: simulates top_k param of retriever.
            remarks: there might be a discrepancy between simulated reader metrics and an actual pipeline run with retriever top_k
        :param doc_relevance_col: column that contains the relevance criteria for documents.
            values can be: 'gold_id_match', 'answer_match', 'gold_id_or_answer_match'
        :param document_metric: the document metric worst queries are calculated with.
            values can be: 'recall_single_hit', 'recall_multi_hit', 'mrr', 'map', 'precision'
        :param document_metric: the answer metric worst queries are calculated with.
            values can be: 'f1', 'exact_match' and 'sas' if the evaluation was made using a SAS model.
        :param eval_mode: the input on which the node was evaluated on.
            Usually nodes get evaluated on the prediction provided by its predecessor nodes in the pipeline (value='integrated').
            However, as the quality of the node itself can heavily depend on the node's input and thus the predecessor's quality,
            you might want to simulate a perfect predecessor in order to get an independent upper bound of the quality of your node.
            For example when evaluating the reader use value='isolated' to simulate a perfect retriever in an ExtractiveQAPipeline.
            Values can be 'integrated', 'isolated'.
            Default value is 'integrated'.
        """
        node_df = self.node_results[node]
        node_df = self._filter_eval_mode(node_df, eval_mode)

        answers = node_df[node_df["type"] == "answer"]
        if len(answers) > 0:
            metrics_df = self._build_answer_metrics_df(
                answers,
                simulated_top_k_reader=simulated_top_k_reader,
                simulated_top_k_retriever=simulated_top_k_retriever,
            )
            worst_df = metrics_df.sort_values(by=[answer_metric]).head(n)
            wrong_examples = []
            for multilabel_id, metrics in worst_df.iterrows():
                query_answers = answers[answers["multilabel_id"] == multilabel_id]
                query_dict = {
                    "multilabel_id": query_answers["multilabel_id"].iloc[0],
                    "query": query_answers["query"].iloc[0],
                    "filters": query_answers["filters"].iloc[0],
                    "metrics": metrics.to_dict(),
                    "answers": query_answers.drop(
                        ["node", "query", "type", "gold_answers", "gold_offsets_in_documents", "gold_document_ids"],
                        axis=1,
                    ).to_dict(orient="records"),
                    "gold_answers": query_answers["gold_answers"].iloc[0],
                    "gold_document_ids": query_answers["gold_document_ids"].iloc[0],
                }
                wrong_examples.append(query_dict)
            return wrong_examples

        documents = node_df[node_df["type"] == "document"]
        if len(documents) > 0:
            metrics_df = self._build_document_metrics_df(
                documents, simulated_top_k_retriever=simulated_top_k_retriever, doc_relevance_col=doc_relevance_col
            )
            worst_df = metrics_df.sort_values(by=[document_metric]).head(n)
            wrong_examples = []
            for multilabel_id, metrics in worst_df.iterrows():
                query_documents = documents[documents["multilabel_id"] == multilabel_id]
                query_dict = {
                    "multilabel_id": query_documents["multilabel_id"].iloc[0],
                    "query": query_documents["query"].iloc[0],
                    "filters": query_documents["filters"].iloc[0],
                    "metrics": metrics.to_dict(),
                    "documents": query_documents.drop(
                        [
                            "node",
                            "query",
                            "multilabel_id",
                            "filters",
                            "type",
                            "gold_document_ids",
                            "gold_document_contents",
                        ],
                        axis=1,
                    ).to_dict(orient="records"),
                    "gold_document_ids": query_documents["gold_document_ids"].iloc[0],
                }
                wrong_examples.append(query_dict)
            return wrong_examples

        return []

    def _calculate_node_metrics(
        self,
        df: pd.DataFrame,
        simulated_top_k_reader: int = -1,
        simulated_top_k_retriever: int = -1,
        doc_relevance_col: str = "gold_id_match",
        eval_mode: str = "integrated",
    ) -> Dict[str, float]:
        df = self._filter_eval_mode(df, eval_mode)

        answer_metrics = self._calculate_answer_metrics(
            df, simulated_top_k_reader=simulated_top_k_reader, simulated_top_k_retriever=simulated_top_k_retriever
        )

        document_metrics = self._calculate_document_metrics(
            df, simulated_top_k_retriever=simulated_top_k_retriever, doc_relevance_col=doc_relevance_col
        )

        return {**answer_metrics, **document_metrics}

    def _filter_eval_mode(self, df: pd.DataFrame, eval_mode: str) -> pd.DataFrame:
        if "eval_mode" in df.columns:
            df = df[df["eval_mode"] == eval_mode]
        else:
            logger.warning("eval dataframe has no eval_mode column. eval_mode param will be ignored.")
        return df

    def _calculate_answer_metrics(
        self, df: pd.DataFrame, simulated_top_k_reader: int = -1, simulated_top_k_retriever: int = -1
    ) -> Dict[str, float]:
        answers = df[df["type"] == "answer"]
        if len(answers) == 0:
            return {}

        metrics_df = self._build_answer_metrics_df(
            answers, simulated_top_k_reader=simulated_top_k_reader, simulated_top_k_retriever=simulated_top_k_retriever
        )

        return {metric: metrics_df[metric].mean() for metric in metrics_df.columns}

    def _build_answer_metrics_df(
        self, answers: pd.DataFrame, simulated_top_k_reader: int = -1, simulated_top_k_retriever: int = -1
    ) -> pd.DataFrame:
        """
        Builds a dataframe containing answer metrics (columns) per query (index).
        Answer metrics are:
        - exact_match (Did the query exactly return any gold answer? -> 1.0 or 0.0)
        - f1 (How well does the best matching returned results overlap with any gold answer on token basis?)
        - sas if a SAS model has bin provided during during pipeline.eval() (How semantically similar is the prediction to the gold answers?)
        """
        # simulate top k retriever
        if simulated_top_k_retriever != -1:
            documents = self._get_documents_df()

            top_k_documents = documents[documents["rank"] <= simulated_top_k_retriever]
            simulated_answers = []
            for multilabel_id in answers["multilabel_id"].unique():
                top_k_document_ids = top_k_documents[top_k_documents["multilabel_id"] == multilabel_id][
                    "document_id"
                ].unique()
                query_answers = answers[answers["multilabel_id"] == multilabel_id]
                # consider only the answers within simulated_top_k_retriever documents
                simulated_query_answers = query_answers[query_answers["document_id"].isin(top_k_document_ids)]
                # simulate top k reader
                if simulated_top_k_reader != -1:
                    # consider only the simulated_top_k_reader answers within simulated_query_answers
                    simulated_query_answers = simulated_query_answers.nsmallest(simulated_top_k_reader, "rank")
                simulated_query_answers["rank"] = np.arange(1, len(simulated_query_answers) + 1)
                simulated_answers.append(simulated_query_answers)
            answers = pd.concat(simulated_answers)
        # simulate top k reader
        elif simulated_top_k_reader != -1:
            answers = answers[answers["rank"] <= simulated_top_k_reader]

        # build metrics df
        metrics = []

        for multilabel_id in answers["multilabel_id"].unique():
            query_df = answers[answers["multilabel_id"] == multilabel_id]

            metrics_cols = set(query_df.columns).intersection(["exact_match", "f1", "sas"])

            query_metrics = {metric: query_df[metric].max() if len(query_df) > 0 else 0.0 for metric in metrics_cols}
            metrics.append(query_metrics)

        metrics_df = pd.DataFrame.from_records(metrics, index=answers["multilabel_id"].unique())
        return metrics_df

    def _get_documents_df(self):
        document_dfs = [
            node_df for node_df in self.node_results.values() if len(node_df[node_df["type"] == "document"]) > 0
        ]
        if len(document_dfs) != 1:
            raise ValueError("cannot detect retriever dataframe")
        documents_df = document_dfs[0]
        documents_df = documents_df[documents_df["type"] == "document"]
        return documents_df

    def _calculate_document_metrics(
        self, df: pd.DataFrame, simulated_top_k_retriever: int = -1, doc_relevance_col: str = "gold_id_match"
    ) -> Dict[str, float]:
        documents = df[df["type"] == "document"]
        if len(documents) == 0:
            return {}

        metrics_df = self._build_document_metrics_df(
            documents, simulated_top_k_retriever=simulated_top_k_retriever, doc_relevance_col=doc_relevance_col
        )

        return {metric: metrics_df[metric].mean() for metric in metrics_df.columns}

    def _build_document_metrics_df(
        self, documents: pd.DataFrame, simulated_top_k_retriever: int = -1, doc_relevance_col: str = "gold_id_match"
    ) -> pd.DataFrame:
        """
        Builds a dataframe containing document metrics (columns) per pair of query and gold document ids (index).
        Document metrics are:
        - mrr (Mean Reciprocal Rank: see https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
        - map (Mean Average Precision: see https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)
        - precision (Precision: How many of the returned documents were relevant?)
        - recall_multi_hit (Recall according to Information Retrieval definition: How many of the relevant documents were retrieved per query?)
        - recall_single_hit (Recall for Question Answering: Did the query return at least one relevant document? -> 1.0 or 0.0)
        """
        if simulated_top_k_retriever != -1:
            documents = documents[documents["rank"] <= simulated_top_k_retriever]

        metrics = []

        for multilabel_id in documents["multilabel_id"].unique():
            query_df = documents[documents["multilabel_id"] == multilabel_id]
            gold_ids = list(query_df["gold_document_ids"].iloc[0])
            retrieved = len(query_df)

            relevance_criteria_ids = list(query_df[query_df[doc_relevance_col] == 1]["document_id"].values)
            num_relevants = len(set(gold_ids + relevance_criteria_ids))
            num_retrieved_relevants = query_df[doc_relevance_col].values.sum()
            rank_retrieved_relevants = query_df[query_df[doc_relevance_col] == 1]["rank"].values
            avp_retrieved_relevants = [
                query_df[doc_relevance_col].values[: int(rank)].sum() / rank for rank in rank_retrieved_relevants
            ]

            avg_precision = np.sum(avp_retrieved_relevants) / num_relevants if num_relevants > 0 else 0.0
            recall_multi_hit = num_retrieved_relevants / num_relevants if num_relevants > 0 else 0.0
            recall_single_hit = min(num_retrieved_relevants, 1)
            precision = num_retrieved_relevants / retrieved if retrieved > 0 else 0.0
            rr = 1.0 / rank_retrieved_relevants.min() if len(rank_retrieved_relevants) > 0 else 0.0
            dcg = (
                np.sum([1.0 / np.log2(rank + 1) for rank in rank_retrieved_relevants])
                if len(rank_retrieved_relevants) > 0
                else 0.0
            )
            idcg = (
                np.sum([1.0 / np.log2(rank + 1) for rank in range(1, num_relevants + 1)]) if num_relevants > 0 else 1.0
            )
            ndcg = dcg / idcg

            metrics.append(
                {
                    "recall_multi_hit": recall_multi_hit,
                    "recall_single_hit": recall_single_hit,
                    "precision": precision,
                    "map": avg_precision,
                    "mrr": rr,
                    "ndcg": ndcg,
                }
            )

        metrics_df = pd.DataFrame.from_records(metrics, index=documents["multilabel_id"].unique())
        return metrics_df

    def save(self, out_dir: Union[str, Path]):
        """
        Saves the evaluation result.
        The result of each node is saved in a separate csv with file name {node_name}.csv to the out_dir folder.

        :param out_dir: Path to the target folder the csvs will be saved.
        """
        out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)
        for node_name, df in self.node_results.items():
            target_path = out_dir / f"{node_name}.csv"
            df.to_csv(target_path, index=False, header=True)

    @classmethod
    def load(cls, load_dir: Union[str, Path]):
        """
        Loads the evaluation result from disk. Expects one csv file per node. See save() for further information.

        :param load_dir: The directory containing the csv files.
        """
        load_dir = load_dir if isinstance(load_dir, Path) else Path(load_dir)
        csv_files = [file for file in load_dir.iterdir() if file.is_file() and file.suffix == ".csv"]
        cols_to_convert = ["gold_document_ids", "gold_document_contents", "gold_answers", "gold_offsets_in_documents"]
        converters = dict.fromkeys(cols_to_convert, ast.literal_eval)
        node_results = {file.stem: pd.read_csv(file, header=0, converters=converters) for file in csv_files}
        result = cls(node_results)
        return result
