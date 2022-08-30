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
from typing import Dict, List, Optional, Any

import copy
import json
import inspect
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from pandas.core.frame import DataFrame
import yaml
from networkx import DiGraph
from networkx.drawing.nx_agraph import to_agraph

from pipelines.pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
)
from pipelines.pipelines.utils import generate_code

try:
    from ray import serve
    import ray
except:
    ray = None  # type: ignore
    serve = None  # type: ignore

try:
    from pipelines import __version__
except:
    # For development
    __version__ = "0.0.0"

from pipelines.schema import Document
from pipelines.nodes.base import BaseComponent
from pipelines.nodes.retriever.base import BaseRetriever
from pipelines.document_stores.base import BaseDocumentStore

logger = logging.getLogger(__name__)

ROOT_NODE_TO_PIPELINE_NAME = {"query": "query", "file": "indexing"}
CODE_GEN_DEFAULT_COMMENT = "This code has been generated."


class RootNode(BaseComponent):
    """
    RootNode feeds inputs together with corresponding params to a Pipeline.
    """

    outgoing_edges = 1

    def run(self, root_node: str):  # type: ignore
        return {}, "output_1"


class BasePipeline:
    """
    Base class for pipelines, providing the most basic methods to load and save them in different ways.
    See also the `Pipeline` class for the actual pipeline logic.
    """

    def run(self, **kwargs):
        raise NotImplementedError

    def get_config(self, return_defaults: bool = False) -> dict:
        """
        Returns a configuration for the Pipeline that can be used with `BasePipeline.load_from_config()`.

        :param return_defaults: whether to output parameters that have the default values.
        """
        raise NotImplementedError

    def to_code(
        self,
        pipeline_variable_name: str = "pipeline",
        generate_imports: bool = True,
        add_comment: bool = False,
    ) -> str:
        """
        Returns the code to create this pipeline as string.

        :param pipeline_variable_name: The variable name of the generated pipeline.
                                       Default value is 'pipeline'.
        :param generate_imports: Whether to include the required import statements into the code.
                                 Default value is True.
        :param add_comment: Whether to add a preceding comment that this code has been generated.
                            Default value is False.
        """
        pipeline_config = self.get_config()
        code = generate_code(
            pipeline_config=pipeline_config,
            pipeline_variable_name=pipeline_variable_name,
            generate_imports=generate_imports,
            comment=CODE_GEN_DEFAULT_COMMENT if add_comment else None,
        )
        return code

    def to_notebook_cell(
        self,
        pipeline_variable_name: str = "pipeline",
        generate_imports: bool = True,
        add_comment: bool = True,
    ):
        """
        Creates a new notebook cell with the code to create this pipeline.

        :param pipeline_variable_name: The variable name of the generated pipeline.
                                       Default value is 'pipeline'.
        :param generate_imports: Whether to include the required import statements into the code.
                                 Default value is True.
        :param add_comment: Whether to add a preceding comment that this code has been generated.
                            Default value is True.
        """
        pipeline_config = self.get_config()
        code = generate_code(
            pipeline_config=pipeline_config,
            pipeline_variable_name=pipeline_variable_name,
            generate_imports=generate_imports,
            comment=CODE_GEN_DEFAULT_COMMENT if add_comment else None,
            add_pipeline_cls_import=False,
        )
        try:
            get_ipython().set_next_input(code)  # type: ignore
        except NameError:
            logger.error(
                "Could not create notebook cell. Make sure you're running in a notebook environment."
            )

    @classmethod
    def load_from_config(cls,
                         pipeline_config: Dict,
                         pipeline_name: Optional[str] = None,
                         overwrite_with_env_variables: bool = True):
        """
        Load Pipeline from a config dict defining the individual components and how they're tied together to form
        a Pipeline. A single config can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

            ```python
            |   {
            |       "version": "1.0",
            |       "components": [
            |           {  # define all the building-blocks for Pipeline
            |               "name": "MyReader",  # custom-name for the component; helpful for visualization & debugging
            |               "type": "FARMReader",  # pipelines Class name for the component
            |               "params": {"no_ans_boost": -10, "model_name_or_path": "ernie-gram-zh-finetuned-dureader-robust"},
            |           },
            |           {
            |               "name": "MyESRetriever",
            |               "type": "ElasticsearchRetriever",
            |               "params": {
            |                   "document_store": "MyDocumentStore",  # params can reference other components defined in the YAML
            |                   "custom_query": None,
            |               },
            |           },
            |           {"name": "MyDocumentStore", "type": "ElasticsearchDocumentStore", "params": {"index": "pipelines_test"}},
            |       ],
            |       "pipelines": [
            |           {  # multiple Pipelines can be defined using the components from above
            |               "name": "my_query_pipeline",  # a simple extractive-qa Pipeline
            |               "nodes": [
            |                   {"name": "MyESRetriever", "inputs": ["Query"]},
            |                   {"name": "MyReader", "inputs": ["MyESRetriever"]},
            |               ],
            |           }
            |       ],
            |   }
            ```

        :param pipeline_config: the pipeline config as dict
        :param pipeline_name: if the config contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        """
        pipeline_definition = get_pipeline_definition(
            pipeline_config=pipeline_config, pipeline_name=pipeline_name)
        if pipeline_definition["type"] == "Pipeline":
            return Pipeline.load_from_config(
                pipeline_config=pipeline_config,
                pipeline_name=pipeline_name,
                overwrite_with_env_variables=overwrite_with_env_variables,
            )
        elif pipeline_definition["type"] == "RayPipeline":
            return RayPipeline.load_from_config(
                pipeline_config=pipeline_config,
                pipeline_name=pipeline_name,
                overwrite_with_env_variables=overwrite_with_env_variables,
            )
        else:
            raise KeyError(
                f"Pipeline Type '{pipeline_definition['type']}' is not a valid. The available types are"
                f"'Pipeline' and 'RayPipeline'.")

    @classmethod
    def load_from_yaml(cls,
                       path: Path,
                       pipeline_name: Optional[str] = None,
                       overwrite_with_env_variables: bool = True):
        """
        Load Pipeline from a YAML file defining the individual components and how they're tied together to form
        a Pipeline. A single YAML can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

            ```yaml
            |   version: '1.0'
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

        Note that, in case of a mismatch in version between pipelines and the YAML, a warning will be printed.
        If the pipeline loads correctly regardless, save again the pipeline using `Pipeline.save_to_yaml()` to remove the warning.

        :param path: path of the YAML file.
        :param pipeline_name: if the YAML contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the YAML configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        """

        pipeline_config = read_pipeline_config_from_yaml(path)
        if pipeline_config["version"] != __version__:
            logger.warning(
                f"YAML version ({pipeline_config['version']}) does not match with pipelines version ({__version__}). "
                "Issues may occur during loading. "
                "To fix this warning, save again this pipeline with the current pipelines version using Pipeline.save_to_yaml(), "
                f"or downgrade to pipelines version {__version__}.")
        return cls.load_from_config(
            pipeline_config=pipeline_config,
            pipeline_name=pipeline_name,
            overwrite_with_env_variables=overwrite_with_env_variables,
        )


class Pipeline(BasePipeline):
    """
    Pipeline brings together building blocks to build a complex search pipeline with pipelines & user-defined components.

    Under-the-hood, a pipeline is represented as a directed acyclic graph of component nodes. It enables custom query
    flows with options to branch queries(eg, extractive qa vs keyword match query), merge candidate documents for a
    Reader from multiple Retrievers, or re-ranking of candidate documents.
    """

    def __init__(self):
        self.graph = DiGraph()
        self.root_node = None

    @property
    def components(self):
        return {
            name: attributes["component"]
            for name, attributes in self.graph.nodes.items()
            if not isinstance(attributes["component"], RootNode)
        }

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
        if self.root_node is None:
            root_node = inputs[0]
            if root_node in ["Query", "File"]:
                self.root_node = root_node
                self.graph.add_node(root_node, component=RootNode())
            else:
                raise KeyError(
                    f"Root node '{root_node}' is invalid. Available options are 'Query' and 'File'."
                )
        component.name = name
        self.graph.add_node(name, component=component, inputs=inputs)

        if len(self.graph.nodes) == 2:  # first node added; connect with Root
            assert len(inputs) == 1 and inputs[0].split(
                ".")[0] == self.root_node, (
                    f"The '{name}' node can only input from {self.root_node}. "
                    f"Set the 'inputs' parameter to ['{self.root_node}']")
            self.graph.add_edge(self.root_node, name, label="output_1")
            return

        for i in inputs:
            if "." in i:
                [input_node_name, input_edge_name] = i.split(".")
                assert "output_" in input_edge_name, f"'{input_edge_name}' is not a valid edge name."
                outgoing_edges_input_node = self.graph.nodes[input_node_name][
                    "component"].outgoing_edges
                assert int(
                    input_edge_name.split("_")[1]
                ) <= outgoing_edges_input_node, (
                    f"Cannot connect '{input_edge_name}' from '{input_node_name}' as it only has "
                    f"{outgoing_edges_input_node} outgoing edge(s).")
            else:
                outgoing_edges_input_node = self.graph.nodes[i][
                    "component"].outgoing_edges
                assert outgoing_edges_input_node == 1, (
                    f"Adding an edge from {i} to {name} is ambiguous as {i} has {outgoing_edges_input_node} edges. "
                    f"Please specify the output explicitly.")
                input_node_name = i
                input_edge_name = "output_1"
            self.graph.add_edge(input_node_name, name, label=input_edge_name)

    def get_node(self, name: str) -> Optional[BaseComponent]:
        """
        Get a node from the Pipeline.

        :param name: The name of the node.
        """
        graph_node = self.graph.nodes.get(name)
        component = graph_node["component"] if graph_node else None
        return component

    def set_node(self, name: str, component):
        """
        Set the component for a node in the Pipeline.

        :param name: The name of the node.
        :param component: The component object to be set at the node.
        """
        self.graph.nodes[name]["component"] = component

    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        """
        Runs the pipeline, one node at a time.

        :param query: The search query (for query pipelines only)
        :param file_paths: The files to index (for indexing pipelines only)
        :param labels:
        :param documents:
        :param meta:
        :param params: Dictionary of parameters to be dispatched to the nodes.
                       If you want to pass a param to all nodes, you can just use: {"top_k":10}
                       If you want to pass it to targeted nodes, you can do:
                       {"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}
        :param debug: Whether the pipeline should instruct nodes to collect debug information
                      about their execution. By default these include the input parameters
                      they received and the output they generated. All debug information can
                      then be found in the dict returned by this method under the key "_debug"
        """
        # validate the node names
        if params:
            if not all(node_id in self.graph.nodes
                       for node_id in params.keys()):

                # Might be a non-targeted param. Verify that too
                not_a_node = set(params.keys()) - set(self.graph.nodes)
                valid_global_params = set()
                for node_id in self.graph.nodes:
                    run_signature_args = inspect.signature(
                        self.graph.nodes[node_id]
                        ["component"].run).parameters.keys()
                    valid_global_params |= set(run_signature_args)
                invalid_keys = [
                    key for key in not_a_node if key not in valid_global_params
                ]

                if invalid_keys:
                    raise ValueError(
                        f"No node(s) or global parameter(s) named {', '.join(invalid_keys)} found in pipeline."
                    )

        node_output = None
        queue = {
            self.root_node: {
                "root_node": self.root_node,
                "params": params
            }
        }  # ordered dict with "node_id" -> "input" mapping that acts as a FIFO queue
        if query:
            queue[self.root_node]["query"] = query
        if file_paths:
            queue[self.root_node]["file_paths"] = file_paths
        if labels:
            queue[self.root_node]["labels"] = labels
        if documents:
            queue[self.root_node]["documents"] = documents
        if meta:
            queue[self.root_node]["meta"] = meta

        i = 0  # the first item is popped off the queue unless it is a "join" node with unprocessed predecessors
        while queue:
            node_id = list(queue.keys())[i]
            node_input = queue[node_id]
            node_input["node_id"] = node_id

            # Apply debug attributes to the node input params
            # NOTE: global debug attributes will override the value specified
            # in each node's params dictionary.
            if debug is not None:
                if node_id not in node_input["params"].keys():
                    node_input["params"][node_id] = {}
                node_input["params"][node_id]["debug"] = debug

            predecessors = set(nx.ancestors(self.graph, node_id))
            if predecessors.isdisjoint(set(queue.keys(
            ))):  # only execute if predecessor nodes are executed
                try:
                    logger.debug(
                        f"Running node `{node_id}` with input `{node_input}`")
                    node_output, stream_id = self.graph.nodes[node_id][
                        "component"]._dispatch_run(**node_input)
                except Exception as e:
                    tb = traceback.format_exc()
                    raise Exception(
                        f"Exception while running node `{node_id}` with input `{node_input}`: {e}, full stack trace: {tb}"
                    )
                queue.pop(node_id)
                #
                if stream_id == "split_documents":
                    for stream_id in [
                            key for key in node_output.keys()
                            if key.startswith("output_")
                    ]:
                        current_node_output = {
                            k: v
                            for k, v in node_output.items()
                            if not k.startswith("output_")
                        }
                        current_docs = node_output.pop(stream_id)
                        current_node_output["documents"] = current_docs
                        next_nodes = self.get_next_nodes(node_id, stream_id)
                        for n in next_nodes:
                            queue[n] = current_node_output
                else:
                    next_nodes = self.get_next_nodes(node_id, stream_id)
                    for n in next_nodes:  # add successor nodes with corresponding inputs to the queue
                        if queue.get(
                                n):  # concatenate inputs if it's a join node
                            existing_input = queue[n]
                            if "inputs" not in existing_input.keys():
                                updated_input: dict = {
                                    "inputs": [existing_input, node_output],
                                    "params": params
                                }
                                if query:
                                    updated_input["query"] = query
                                if file_paths:
                                    updated_input["file_paths"] = file_paths
                                if labels:
                                    updated_input["labels"] = labels
                                if documents:
                                    updated_input["documents"] = documents
                                if meta:
                                    updated_input["meta"] = meta
                            else:
                                existing_input["inputs"].append(node_output)
                                updated_input = existing_input
                            queue[n] = updated_input
                        else:
                            queue[n] = node_output
                i = 0
            else:
                i += 1  # attempt executing next node in the queue as current `node_id` has unprocessed predecessors
        return node_output

    def _reorder_columns(self, df: DataFrame,
                         desired_order: List[str]) -> DataFrame:
        filtered_order = [col for col in desired_order if col in df.columns]
        missing_columns = [
            col for col in df.columns if col not in desired_order
        ]
        reordered_columns = filtered_order + missing_columns
        assert len(reordered_columns) == len(df.columns)
        return df.reindex(columns=reordered_columns)

    def _build_eval_dataframe(self, query: str, query_labels: MultiLabel,
                              node_name: str, node_output: dict) -> DataFrame:
        """
        Builds a Dataframe for each query from which evaluation metrics can be calculated.
        Currently only answer or document returning nodes are supported, returns None otherwise.

        Each row contains either an answer or a document that has been retrieved during evaluation.
        Rows are being enriched with basic infos like rank, query, type or node.
        Additional answer or document specific evaluation infos like gold labels
        and metrics depicting whether the row matches the gold labels are included, too.
        """

        if query_labels is None or query_labels.labels is None:
            logger.warning(
                f"There is no label for query '{query}'. Query will be omitted."
            )
            return pd.DataFrame()

        # remarks for no_answers:
        # Single 'no_answer'-labels are not contained in MultiLabel aggregates.
        # If all labels are no_answers, MultiLabel.answers will be [""] and the other aggregates []
        gold_answers = query_labels.answers
        gold_offsets_in_documents = query_labels.gold_offsets_in_documents
        gold_document_ids = query_labels.document_ids
        gold_document_contents = query_labels.document_contents

        # if node returned answers, include answer specific info:
        # - the answer returned itself
        # - the document_id the answer was found in
        # - the position or offsets within the document the answer was found
        # - the surrounding context of the answer within the document
        # - the gold answers
        # - the position or offsets of the gold answer within the document
        # - the gold document ids containing the answer
        # - the exact_match metric depicting if the answer exactly matches the gold label
        # - the f1 metric depicting how well the answer overlaps with the gold label on token basis
        # - the sas metric depicting how well the answer matches the gold label on a semantic basis.
        #   this will be calculated on all queries in eval() for performance reasons if a sas model has been provided

        partial_dfs = []
        for field_name in ["answers", "answers_isolated"]:
            df = pd.DataFrame()
            answers = node_output.get(field_name, None)
            if answers is not None:
                answer_cols_to_keep = [
                    "answer", "document_id", "offsets_in_document", "context"
                ]
                df_answers = pd.DataFrame(answers, columns=answer_cols_to_keep)
                if len(df_answers) > 0:
                    df_answers["type"] = "answer"
                    df_answers["gold_answers"] = [gold_answers
                                                  ] * len(df_answers)
                    df_answers["gold_offsets_in_documents"] = [
                        gold_offsets_in_documents
                    ] * len(df_answers)
                    df_answers["gold_document_ids"] = [gold_document_ids
                                                       ] * len(df_answers)
                    df_answers["exact_match"] = df_answers.apply(
                        lambda row: calculate_em_str_multi(
                            gold_answers, row["answer"]),
                        axis=1)
                    df_answers["f1"] = df_answers.apply(
                        lambda row: calculate_f1_str_multi(
                            gold_answers, row["answer"]),
                        axis=1)
                    df_answers["rank"] = np.arange(1, len(df_answers) + 1)
                    df = pd.concat([df, df_answers])

            # add general info
            df["node"] = node_name
            df["multilabel_id"] = query_labels.id
            df["query"] = query
            df["filters"] = json.dumps(query_labels.filters,
                                       sort_keys=True).encode()
            df["eval_mode"] = "isolated" if "isolated" in field_name else "integrated"
            partial_dfs.append(df)

        # if node returned documents, include document specific info:
        # - the document_id
        # - the content of the document
        # - the gold document ids
        # - the gold document contents
        # - the gold_id_match metric depicting whether one of the gold document ids matches the document
        # - the answer_match metric depicting whether the document contains the answer
        # - the gold_id_or_answer_match metric depicting whether one of the former two conditions are met
        for field_name in ["documents", "documents_isolated"]:
            df = pd.DataFrame()
            documents = node_output.get(field_name, None)
            if documents is not None:
                document_cols_to_keep = ["content", "id"]
                df_docs = pd.DataFrame(documents, columns=document_cols_to_keep)
                if len(df_docs) > 0:
                    df_docs = df_docs.rename(columns={"id": "document_id"})
                    df_docs["type"] = "document"
                    df_docs["gold_document_ids"] = [gold_document_ids
                                                    ] * len(df_docs)
                    df_docs["gold_document_contents"] = [
                        gold_document_contents
                    ] * len(df_docs)
                    df_docs["gold_id_match"] = df_docs.apply(
                        lambda row: 1.0
                        if row["document_id"] in gold_document_ids else 0.0,
                        axis=1)
                    df_docs["answer_match"] = df_docs.apply(
                        lambda row: 1.0 if not query_labels.no_answer and any(
                            gold_answer in row["content"]
                            for gold_answer in gold_answers) else 0.0,
                        axis=1,
                    )
                    df_docs["gold_id_or_answer_match"] = df_docs.apply(
                        lambda row: max(row["gold_id_match"], row["answer_match"
                                                                  ]),
                        axis=1)
                    df_docs["rank"] = np.arange(1, len(df_docs) + 1)
                    df = pd.concat([df, df_docs])

            # add general info
            df["node"] = node_name
            df["multilabel_id"] = query_labels.id
            df["query"] = query
            df["filters"] = json.dumps(query_labels.filters,
                                       sort_keys=True).encode()
            df["eval_mode"] = "isolated" if "isolated" in field_name else "integrated"
            partial_dfs.append(df)

        return pd.concat(partial_dfs, ignore_index=True)

    def get_next_nodes(self, node_id: str, stream_id: str):
        current_node_edges = self.graph.edges(node_id, data=True)
        next_nodes = [
            next_node for _, next_node, data in current_node_edges
            if not stream_id or data["label"] == stream_id
            or stream_id == "output_all"
        ]
        return next_nodes

    def get_nodes_by_class(self, class_type) -> List[Any]:
        """
        Gets all nodes in the pipeline that are an instance of a certain class (incl. subclasses).
        This is for example helpful if you loaded a pipeline and then want to interact directly with the document store.
        Example:
        | from pipelines.document_stores.base import BaseDocumentStore
        | INDEXING_PIPELINE = Pipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME)
        | res = INDEXING_PIPELINE.get_nodes_by_class(class_type=BaseDocumentStore)

        :return: List of components that are an instance the requested class
        """

        matches = [
            self.graph.nodes.get(node)["component"] for node in self.graph.nodes
            if isinstance(self.graph.nodes.get(node)["component"], class_type)
        ]
        return matches

    def get_document_store(self) -> Optional[BaseDocumentStore]:
        """
        Return the document store object used in the current pipeline.

        :return: Instance of DocumentStore or None
        """
        matches = self.get_nodes_by_class(class_type=BaseDocumentStore)
        if len(matches) == 0:
            matches = list(
                set(retriever.document_store
                    for retriever in self.get_nodes_by_class(
                        class_type=BaseRetriever)))

        if len(matches) > 1:
            raise Exception(
                f"Multiple Document Stores found in Pipeline: {matches}")
        if len(matches) == 0:
            return None
        else:
            return matches[0]

    def draw(self, path: Path = Path("pipeline.png")):
        """
        Create a Graphviz visualization of the pipeline.

        :param path: the path to save the image.
        """
        try:
            import pygraphviz
        except ImportError:
            raise ImportError(
                f"Could not import `pygraphviz`. Please install via: \n"
                f"pip install pygraphviz\n"
                f"(You might need to run this first: apt install libgraphviz-dev graphviz )"
            )

        graphviz = to_agraph(self.graph)
        graphviz.layout("dot")
        graphviz.draw(path)

    @classmethod
    def load_from_config(cls,
                         pipeline_config: Dict,
                         pipeline_name: Optional[str] = None,
                         overwrite_with_env_variables: bool = True):
        """
        Load Pipeline from a config dict defining the individual components and how they're tied together to form
        a Pipeline. A single config can declare multiple Pipelines, in which case an explicit `pipeline_name` must
        be passed.

        Here's a sample configuration:

            ```python
            |   {
            |       "version": "0.9",
            |       "components": [
            |           {  # define all the building-blocks for Pipeline
            |               "name": "MyReader",  # custom-name for the component; helpful for visualization & debugging
            |               "type": "FARMReader",  # pipelines Class name for the component
            |               "params": {"no_ans_boost": -10, "model_name_or_path": "ernie-gram-zh-finetuned-dureader-robust"},
            |           },
            |           {
            |               "name": "MyESRetriever",
            |               "type": "ElasticsearchRetriever",
            |               "params": {
            |                   "document_store": "MyDocumentStore",  # params can reference other components defined in the YAML
            |                   "custom_query": None,
            |               },
            |           },
            |           {"name": "MyDocumentStore", "type": "ElasticsearchDocumentStore", "params": {"index": "pipelines_test"}},
            |       ],
            |       "pipelines": [
            |           {  # multiple Pipelines can be defined using the components from above
            |               "name": "my_query_pipeline",  # a simple extractive-qa Pipeline
            |               "nodes": [
            |                   {"name": "MyESRetriever", "inputs": ["Query"]},
            |                   {"name": "MyReader", "inputs": ["MyESRetriever"]},
            |               ],
            |           }
            |       ],
            |   }
            ```

        :param pipeline_config: the pipeline config as dict
        :param pipeline_name: if the config contains multiple pipelines, the pipeline_name to load must be set.
        :param overwrite_with_env_variables: Overwrite the configuration with environment variables. For example,
                                             to change index name param for an ElasticsearchDocumentStore, an env
                                             variable 'MYDOCSTORE_PARAMS_INDEX=documents-2021' can be set. Note that an
                                             `_` sign must be used to specify nested hierarchical properties.
        """
        pipeline_definition = get_pipeline_definition(
            pipeline_config=pipeline_config, pipeline_name=pipeline_name)
        component_definitions = get_component_definitions(
            pipeline_config=pipeline_config,
            overwrite_with_env_variables=overwrite_with_env_variables)

        pipeline = cls()

        components: dict = {}  # instances of component objects.
        for node in pipeline_definition["nodes"]:
            name = node["name"]
            component = cls._load_or_get_component(
                name=name,
                definitions=component_definitions,
                components=components)
            pipeline.add_node(component=component,
                              name=name,
                              inputs=node.get("inputs", []))

        return pipeline

    @classmethod
    def _load_or_get_component(cls, name: str, definitions: dict,
                               components: dict):
        """
        Load a component from the definition or return if component object already present in `components` dict.

        :param name: name of the component to load or get.
        :param definitions: dict containing definitions of all components retrieved from the YAML.
        :param components: dict containing component objects.
        """
        try:
            if name in components.keys(
            ):  # check if component is already loaded.
                return components[name]

            component_params = definitions[name].get("params", {})
            component_type = definitions[name]["type"]
            logger.debug(
                f"Loading component `{name}` of type `{definitions[name]['type']}`"
            )

            for key, value in component_params.items():
                # Component params can reference to other components. For instance, a Retriever can reference a
                # DocumentStore defined in the YAML. All references should be recursively resolved.
                if (
                        isinstance(value, str) and value in definitions.keys()
                ):  # check if the param value is a reference to another component.
                    if value not in components.keys(
                    ):  # check if the referenced component is already loaded.
                        cls._load_or_get_component(name=value,
                                                   definitions=definitions,
                                                   components=components)
                    component_params[key] = components[
                        value]  # substitute reference (string) with the component object.

            instance = BaseComponent.load_from_args(
                component_type=component_type, **component_params)
            components[name] = instance
        except Exception as e:
            raise Exception(f"Failed loading pipeline component '{name}': {e}")
        return instance

    def save_to_yaml(self, path: Path, return_defaults: bool = False):
        """
        Save a YAML configuration for the Pipeline that can be used with `Pipeline.load_from_yaml()`.

        :param path: path of the output YAML file.
        :param return_defaults: whether to output parameters that have the default values.
        """
        config = self.get_config(return_defaults=return_defaults)
        with open(path, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def get_config(self, return_defaults: bool = False) -> dict:
        """
        Returns a configuration for the Pipeline that can be used with `Pipeline.load_from_config()`.

        :param return_defaults: whether to output parameters that have the default values.
        """
        pipeline_name = ROOT_NODE_TO_PIPELINE_NAME[self.root_node.lower()]
        pipelines: dict = {
            pipeline_name: {
                "name": pipeline_name,
                "type": self.__class__.__name__,
                "nodes": []
            }
        }

        components = {}
        for node in self.graph.nodes:
            if node == self.root_node:
                continue
            component_instance = self.graph.nodes.get(node)["component"]
            component_type = component_instance.pipeline_config["type"]
            component_params = component_instance.pipeline_config["params"]
            components[node] = {
                "name": node,
                "type": component_type,
                "params": {}
            }

            component_parent_classes = inspect.getmro(type(component_instance))
            component_signature: dict = {}
            for component_parent in component_parent_classes:
                component_signature = {
                    **component_signature,
                    **inspect.signature(component_parent).parameters
                }

            for param_key, param_value in component_params.items():
                # A parameter for a Component could be another Component. For instance, a Retriever has
                # the DocumentStore as a parameter.
                # Component configs must be a dict with a "type" key. The "type" keys distinguishes between
                # other parameters like "custom_mapping" that are dicts.
                # This currently only checks for the case single-level nesting case, wherein, "a Component has another
                # Component as a parameter". For deeper nesting cases, this function should be made recursive.
                if isinstance(param_value, dict) and "type" in param_value.keys(
                ):  # the parameter is a Component
                    sub_component = param_value
                    sub_component_type_name = sub_component["type"]
                    sub_component_signature = inspect.signature(
                        BaseComponent.subclasses[sub_component_type_name]
                    ).parameters
                    sub_component_params = {
                        k: v
                        for k, v in sub_component["params"].items()
                        if sub_component_signature[k].default != v
                        or return_defaults is True
                    }

                    sub_component_name = self._generate_component_name(
                        type_name=sub_component_type_name,
                        params=sub_component_params,
                        existing_components=components)
                    components[sub_component_name] = {
                        "name": sub_component_name,
                        "type": sub_component_type_name,
                        "params": sub_component_params,
                    }
                    components[node]["params"][param_key] = sub_component_name
                else:
                    if component_signature[
                            param_key].default != param_value or return_defaults is True:
                        components[node]["params"][param_key] = param_value

            # create the Pipeline definition with how the Component are connected
            pipelines[pipeline_name]["nodes"].append({
                "name":
                node,
                "inputs":
                list(self.graph.predecessors(node))
            })

        config = {
            "components": list(components.values()),
            "pipelines": list(pipelines.values()),
            "version": __version__,
        }
        return config

    def _generate_component_name(
        self,
        type_name: str,
        params: Dict[str, Any],
        existing_components: Dict[str, Any],
    ):
        component_name: str = type_name
        # add number if there are multiple distinct ones of the same type
        while component_name in existing_components and params != existing_components[
                component_name]["params"]:
            occupied_num = 1
            if len(component_name) > len(type_name):
                occupied_num = int(component_name[len(type_name) + 1:])
            new_num = occupied_num + 1
            component_name = f"{type_name}_{new_num}"
        return component_name
