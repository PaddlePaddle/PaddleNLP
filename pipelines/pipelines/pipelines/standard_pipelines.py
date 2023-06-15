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

import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipelines.document_stores import BaseDocumentStore
from pipelines.nodes.answer_extractor import AnswerExtractor, QAFilter
from pipelines.nodes.base import BaseComponent
from pipelines.nodes.prompt import PromptNode, Shaper
from pipelines.nodes.question_generator import QuestionGenerator
from pipelines.nodes.ranker import BaseRanker, ErnieRanker
from pipelines.nodes.reader import BaseReader
from pipelines.nodes.retriever import BaseRetriever, WebRetriever
from pipelines.nodes.text_to_image_generator import ErnieTextToImageGenerator
from pipelines.pipelines import Pipeline
from pipelines.schema import Answer, Document

logger = logging.getLogger(__name__)


class BaseStandardPipeline(ABC):
    """
    Base class for pre-made standard pipelines pipelines.
    This class does not inherit from Pipeline.
    """

    pipeline: Pipeline
    metrics_filter: Optional[Dict[str, List[str]]] = None

    def add_node(self, component, name: str, inputs: List[str]):
        """
        Add a new node to the pipeline.

        :param component: The object to be called when the data is passed to the node. It can be a pipelines component
                          (like Retriever, Reader, or Generator) or a user-defined object that implements a run()
                          method to process incoming data from predecessor node.
        :param name: The name for the node. It must not contain any dots.
        :param inputs: A list of inputs to the node. If the predecessor node has a single outgoing edge, just the name
                       of node is sufficient. For instance, a 'ElasticsearchRetriever' node would always output a single
                       edge with a list of documents. It can be represented as ["ElasticsearchRetriever"].

                       In cases when the predecessor node has multiple outputs, e.g., a "QueryClassifier", the output
                       must be specified explicitly as "QueryClassifier.output_2".
        """
        self.pipeline.add_node(component=component, name=name, inputs=inputs)

    def get_node(self, name: str):
        """
        Get a node from the Pipeline.

        :param name: The name of the node.
        """
        component = self.pipeline.get_node(name)
        return component

    def set_node(self, name: str, component):
        """
        Set the component for a node in the Pipeline.

        :param name: The name of the node.
        :param component: The component object to be set at the node.
        """
        self.pipeline.set_node(name, component)

    def draw(self, path: Path = Path("pipeline.png")):
        """
        Create a Graphviz visualization of the pipeline.

        :param path: the path to save the image.
        """
        self.pipeline.draw(path)

    def save_to_yaml(self, path: Path, return_defaults: bool = False):
        """
        Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

        :param path: path of the output YAML file.
        :param return_defaults: whether to output parameters that have the default values.
        """
        return self.pipeline.save_to_yaml(path, return_defaults)

    @classmethod
    def load_from_yaml(
        cls, path: Path, pipeline_name: Optional[str] = None, overwrite_with_env_variables: bool = True
    ):
        """
        Load Pipeline from a YAML file defining the individual components and how they're tied together to form
        a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

            ```yaml
            |   version: '0.8'
            |
            |    components:    # define all the building-blocks for Pipeline
            |    - name: MyReader       # custom-name for the component; helpful for visualization & debugging
            |      type: FARMReader    # pipelines Class name for the component
            |      params:
            |        no_ans_boost: -10
            |        model_name_or_path: ernie-gram-zh-finetuned-dureader-robust
            |    - name: MyESRetriever
            |      type: ElasticsearchRetriever
            |      params:
            |        document_store: MyDocumentStore    # params can reference other components defined in the YAML
            |        custom_query: null
            |    - name: MyDocumentStore
            |      type: ElasticsearchDocumentStore
            |      params:
            |        index: pipelines_test
            |
            |    pipelines:    # multiple Pipelines can be defined using the components from above
            |    - name: my_query_pipeline    # a simple extractive-qa Pipeline
            |      nodes:
            |      - name: MyESRetriever
            |        inputs: [Query]
            |      - name: MyReader
            |        inputs: [MyESRetriever]
            ```

        :param path: path of the YAML file.
        :param pipeline_name: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the YAML configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        """
        standard_pipeline_object = cls.__new__(
            cls
        )  # necessary because we can't call __init__ as we can't provide parameters
        standard_pipeline_object.pipeline = Pipeline.load_from_yaml(path, pipeline_name, overwrite_with_env_variables)
        return standard_pipeline_object

    def get_nodes_by_class(self, class_type) -> List[Any]:
        """
        Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).
        This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
        Example:
        ```python
        | from pipelines.document_stores.base import BaseDocumentStore
        | INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
        | res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)
        ```
        :return: List of components that are an instance of the requested class
        """
        return self.pipeline.get_nodes_by_class(class_type)

    def get_document_store(self) -> Optional[BaseDocumentStore]:
        """
        Return the document store object used in the current pipeline.

        :return: Instance of DocumentStore or None
        """
        return self.pipeline.get_document_store()

    def run_batch(self, queries: List[str], params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        Run a batch of queries through the pipeline.
        :param queries: List of query strings.
        :param params: Parameters for the individual nodes of the pipeline. For instance,
                       `params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}`
        :param debug: Whether the pipeline should instruct nodes to collect debug information
                      about their execution. By default these include the input parameters
                      they received and the output they generated.
                      All debug information can then be found in the dict returned
                      by this method under the key "_debug"
        """
        output = self.pipeline.run_batch(queries=queries, params=params, debug=debug)
        return output


class ExtractiveQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Extractive Question Answering.
    """

    def __init__(self, reader: BaseReader, ranker: BaseRanker, retriever: BaseRetriever):
        """
        :param reader: Reader instance
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
        self.pipeline.add_node(component=reader, name="Reader", inputs=["Ranker"])
        self.metrics_filter = {"Retriever": ["recall_single_hit"]}

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: The search query string.
        :param params: Params for the `retriever` and `reader`. For instance,
                       params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
                      about their execution. By default these include the input parameters
                      they received and the output they generated.
                      All debug information can then be found in the dict returned
                      by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class SemanticSearchPipeline(BaseStandardPipeline):
    """
    Pipeline for semantic search.
    """

    def __init__(self, retriever: BaseRetriever, ranker: Optional[BaseRanker] = None):
        """
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        if ranker:
            self.pipeline.add_node(component=ranker, name="Ranker", inputs=["Retriever"])

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output


class DocPipeline(BaseStandardPipeline):
    """
    Pipeline for document intelligence.
    """

    def __init__(self, preprocessor: BaseComponent, docreader: BaseComponent):
        """
        :param preprocessor: file/image preprocessor instance
        :param docreader: document model runner instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["Query"])
        self.pipeline.add_node(component=docreader, name="Reader", inputs=["PreProcessor"])

    def run(self, meta: dict, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(meta=meta, params=params, debug=debug)
        return output


class TextToImagePipeline(BaseStandardPipeline):
    """
    A simple pipeline that takes prompt texts as input and generates
    images.
    """

    def __init__(self, text_to_image_generator: ErnieTextToImageGenerator):
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=text_to_image_generator, name="TextToImageGenerator", inputs=["Query"])

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        output = self.pipeline.run(query=query, params=params, debug=debug)
        return output

    def run_batch(
        self,
        documents: List[Document],
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        output = self.pipeline.run_batch(documents=documents, params=params, debug=debug)
        return output


class QAGenerationPipeline(BaseStandardPipeline):
    """
    Pipeline for semantic search.
    """

    def __init__(self, answer_extractor: AnswerExtractor, question_generator: QuestionGenerator, qa_filter: QAFilter):
        """
        :param retriever: Retriever instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=answer_extractor, name="AnswerExtractor", inputs=["Query"])
        self.pipeline.add_node(component=question_generator, name="QuestionGenerator", inputs=["AnswerExtractor"])
        self.pipeline.add_node(component=qa_filter, name="QAFilter", inputs=["QuestionGenerator"])

    def run(self, meta: List[str], params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(meta=meta, params=params, debug=debug)
        return output


class SentaPipeline(BaseStandardPipeline):
    """
    Pipeline for document intelligence.
    """

    def __init__(self, preprocessor: BaseComponent, senta: BaseComponent, visualization: BaseComponent):
        """
        :param preprocessor: file preprocessor instance
        :param senta: senta model instance
        """
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["File"])
        self.pipeline.add_node(component=senta, name="Senta", inputs=["PreProcessor"])
        self.pipeline.add_node(component=visualization, name="Visualization", inputs=["Senta"])

    def run(self, meta: dict, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: the query string.
        :param params: params for the `retriever` and `reader`. For instance, params={"Retriever": {"top_k": 10}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
              about their execution. By default these include the input parameters
              they received and the output they generated.
              All debug information can then be found in the dict returned
              by this method under the key "_debug"
        """
        output = self.pipeline.run(meta=meta, params=params, debug=debug)
        if "examples" in output:
            output.pop("examples")
        return output


class WebQAPipeline(BaseStandardPipeline):
    """
    Pipeline for Generative Question Answering performed based on Documents returned from a web search engine.
    """

    def __init__(
        self,
        retriever: WebRetriever,
        prompt_node: PromptNode,
        sampler: Optional[BaseRanker] = None,
        shaper: Optional[Shaper] = None,
    ):
        """
        :param retriever: The WebRetriever used for retrieving documents from a web search engine.
        :param prompt_node: The PromptNode used for generating the answer based on retrieved documents.
        :param shaper: The Shaper used for transforming the documents and scores into a format that can be used by the PromptNode. Optional.
        """
        if not shaper:
            shaper = Shaper(func="join_documents_and_scores", inputs={"documents": "documents"}, outputs=["documents"])
        if not sampler and retriever.mode != "snippets":
            # Documents returned by WebRetriever in mode "snippets" already have scores.
            # For other modes, we need to add a sampler if none is provided to compute the scores.
            # TODO(wugaosheng): Add topsampler into WebQAPipeline
            sampler = ErnieRanker("rocketqa-zh-dureader-cross-encoder", top_k=2)

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        if sampler:
            self.pipeline.add_node(component=sampler, name="Sampler", inputs=["Retriever"])
            self.pipeline.add_node(component=shaper, name="Shaper", inputs=["Sampler"])
        else:
            self.pipeline.add_node(component=shaper, name="Shaper", inputs=["Retriever"])
        self.pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Shaper"])
        self.metrics_filter = {"Retriever": ["recall_single_hit"]}

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        """
        :param query: The search query string.
        :param params: Params for the `Retriever`, `Sampler`, `Shaper`, and ``PromptNode. For instance,
                       params={"Retriever": {"top_k": 3}, "Sampler": {"top_p": 0.8}}. See the API documentation of each node for available parameters and their descriptions.
        :param debug: Whether the pipeline should instruct nodes to collect debug information
                      about their execution. By default, these include the input parameters
                      they received and the output they generated.
                      YOu can then find all debug information in the dict thia method returns
                      under the key "_debug".
        """
        output = self.pipeline.run(query=query, params=params, debug=debug)
        # Extract the answer from the last line of the PromptNode's output
        output["answers"] = [Answer(answer=output["results"][0].split("\n")[-1], type="generative")]
        return output
