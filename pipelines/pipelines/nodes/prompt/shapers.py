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

import inspect
import logging
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pipelines.nodes.base import BaseComponent
from pipelines.schema import Answer, Document, MultiLabel

logger = logging.getLogger(__name__)


def rename(value: Any) -> Any:
    """
    An identity function. You can use it to rename values in the invocation context without changing them.

    Example:

    ```python
    assert rename(1) == 1
    ```
    """
    return value


def strings_to_answers(
    strings: List[str],
    prompts: Optional[List[Union[str, List[Dict[str, str]]]]] = None,
    documents: Optional[List[Document]] = None,
    pattern: Optional[str] = None,
    reference_pattern: Optional[str] = None,
    reference_mode: Literal["index", "id", "meta"] = "index",
    reference_meta_field: Optional[str] = None,
) -> List[Answer]:
    """
    Transforms a list of strings into a list of answers.
    Specify `reference_pattern` to populate the answer's `document_ids` by extracting document references from the strings.

    :param strings: The list of strings to transform.
    :param prompts: The prompts used to generate the answers.
    :param documents: The documents used to generate the answers.
    :param pattern: The regex pattern to use for parsing the answer.
        Examples:
            `[^\\n]+$` will find "this is an answer" in string "this is an argument.\nthis is an answer".
            `Answer: (.*)` will find "this is an answer" in string "this is an argument. Answer: this is an answer".
        If None, the whole string is used as the answer. If not None, the first group of the regex is used as the answer. If there is no group, the whole match is used as the answer.
    :param reference_pattern: The regex pattern to use for parsing the document references.
        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".
        If None, no parsing is done and all documents are referenced.
    :param reference_mode: The mode used to reference documents. Supported modes are:
        - index: the document references are the one-based index of the document in the list of documents.
            Example: "this is an answer[1]" will reference the first document in the list of documents.
        - id: the document references are the document IDs.
            Example: "this is an answer[123]" will reference the document with id "123".
        - meta: the document references are the value of a metadata field of the document.
            Example: "this is an answer[123]" will reference the document with the value "123" in the metadata field specified by reference_meta_field.
    :param reference_meta_field: The name of the metadata field to use for document references in reference_mode "meta".
    :return: The list of answers.

    Examples:

    Without reference parsing:
    ```python
    assert strings_to_answers(strings=["first", "second", "third"], prompt="prompt", documents=[Document(id="123", content="content")]) == [
            Answer(answer="first", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
            Answer(answer="second", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
            Answer(answer="third", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
        ]
    ```

    With reference parsing:
    ```python
    assert strings_to_answers(strings=["first[1]", "second[2]", "third[1][3]"], prompt="prompt",
            documents=[Document(id="123", content="content"), Document(id="456", content="content"), Document(id="789", content="content")],
            reference_pattern=r"\\[(\\d+)\\]",
            reference_mode="index"
        ) == [
            Answer(answer="first", type="generative", document_ids=["123"], meta={"prompt": "prompt"}),
            Answer(answer="second", type="generative", document_ids=["456"], meta={"prompt": "prompt"}),
            Answer(answer="third", type="generative", document_ids=["123", "789"], meta={"prompt": "prompt"}),
        ]
    ```
    """
    if prompts:
        if len(prompts) == 1:
            # one prompt for all strings/documents
            documents_per_string: List[Optional[List[Document]]] = [documents] * len(strings)
            prompt_per_string: List[Optional[Union[str, List[Dict[str, str]]]]] = [prompts[0]] * len(strings)
        elif len(prompts) > 1 and len(strings) % len(prompts) == 0:
            # one prompt per string/document
            if documents is not None and len(documents) != len(prompts):
                raise ValueError("The number of documents must match the number of prompts.")
            string_multiplier = len(strings) // len(prompts)
            documents_per_string = (
                [[doc] for doc in documents for _ in range(string_multiplier)] if documents else [None] * len(strings)
            )
            prompt_per_string = [prompt for prompt in prompts for _ in range(string_multiplier)]
        else:
            raise ValueError("The number of prompts must be one or a multiple of the number of strings.")
    else:
        documents_per_string = [documents] * len(strings)
        prompt_per_string = [None] * len(strings)

    answers = []
    for string, prompt, _documents in zip(strings, prompt_per_string, documents_per_string):
        answer = string_to_answer(
            string=string,
            prompt=prompt,
            documents=_documents,
            pattern=pattern,
            reference_pattern=reference_pattern,
            reference_mode=reference_mode,
            reference_meta_field=reference_meta_field,
        )
        answers.append(answer)
    return answers


def string_to_answer(
    string: str,
    prompt: Optional[Union[str, List[Dict[str, str]]]],
    documents: Optional[List[Document]],
    pattern: Optional[str] = None,
    reference_pattern: Optional[str] = None,
    reference_mode: Literal["index", "id", "meta"] = "index",
    reference_meta_field: Optional[str] = None,
) -> Answer:
    """
    Transforms a string into an answer.
    Specify `reference_pattern` to populate the answer's `document_ids` by extracting document references from the string.

    :param string: The string to transform.
    :param prompt: The prompt used to generate the answer.
    :param documents: The documents used to generate the answer.
    :param pattern: The regex pattern to use for parsing the answer.
        Examples:
            `[^\\n]+$` will find "this is an answer" in string "this is an argument.\nthis is an answer".
            `Answer: (.*)` will find "this is an answer" in string "this is an argument. Answer: this is an answer".
        If None, the whole string is used as the answer. If not None, the first group of the regex is used as the answer. If there is no group, the whole match is used as the answer.
    :param reference_pattern: The regex pattern to use for parsing the document references.
        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".
        If None, no parsing is done and all documents are referenced.
    :param reference_mode: The mode used to reference documents. Supported modes are:
        - index: the document references are the one-based index of the document in the list of documents.
            Example: "this is an answer[1]" will reference the first document in the list of documents.
        - id: the document references are the document IDs.
            Example: "this is an answer[123]" will reference the document with id "123".
        - meta: the document references are the value of a metadata field of the document.
            Example: "this is an answer[123]" will reference the document with the value "123" in the metadata field specified by reference_meta_field.
    :param reference_meta_field: The name of the metadata field to use for document references in reference_mode "meta".
    :return: The answer
    """
    if reference_mode == "index":
        candidates = {str(idx): doc.id for idx, doc in enumerate(documents, start=1)} if documents else {}
    elif reference_mode == "id":
        candidates = {doc.id: doc.id for doc in documents} if documents else {}
    elif reference_mode == "meta":
        if not reference_meta_field:
            raise ValueError("reference_meta_field must be specified when reference_mode is 'meta'")
        candidates = (
            {doc.meta[reference_meta_field]: doc.id for doc in documents if doc.meta.get(reference_meta_field)}
            if documents
            else {}
        )
    else:
        raise ValueError(f"Invalid document_id_mode: {reference_mode}")

    if pattern:
        match = re.search(pattern, string)
        if match:
            if not match.lastindex:
                # no group in pattern -> take the whole match
                string = match.group(0)
            elif match.lastindex == 1:
                # one group in pattern -> take the group
                string = match.group(1)
            else:
                # more than one group in pattern -> raise error
                raise ValueError(f"Pattern must have at most one group: {pattern}")
        else:
            string = ""
    document_ids = parse_references(string=string, reference_pattern=reference_pattern, candidates=candidates)
    answer = Answer(answer=string, type="generative", document_ids=document_ids, meta={"prompt": prompt})
    return answer


def parse_references(
    string: str, reference_pattern: Optional[str] = None, candidates: Optional[Dict[str, str]] = None
) -> Optional[List[str]]:
    """
    Parses an answer string for document references and returns the document IDs of the referenced documents.

    :param string: The string to parse.
    :param reference_pattern: The regex pattern to use for parsing the document references.
        Example: `\\[(\\d+)\\]` will find "1" in string "this is an answer[1]".
        If None, no parsing is done and all candidate document IDs are returned.
    :param candidates: A dictionary of candidates to choose from. The keys are the reference strings and the values are the document IDs.
        If None, no parsing is done and None is returned.
    :return: A list of document IDs.
    """
    if not candidates:
        return None
    if not reference_pattern:
        return list(candidates.values())

    document_idxs = re.findall(reference_pattern, string)
    return [candidates[idx] for idx in document_idxs if idx in candidates]


def join_documents_and_scores(documents: List[Document]) -> Tuple[List[Document]]:
    """
    Transforms a list of documents with scores in their metadata into a list containing a single document.
    The resulting document contains the scores and the contents of all the original documents.
    All metadata is dropped.
    Example:
    ```python
    assert join_documents_and_scores(
        documents=[
            Document(content="first", meta={"score": 0.9}),
            Document(content="second", meta={"score": 0.7}),
            Document(content="third", meta={"score": 0.5})
        ],
        delimiter=" - "
    ) == ([Document(content="-[0.9] first\n -[0.7] second\n -[0.5] third")], )
    ```
    """
    content = "\n".join([f"-[{round(float(doc.meta['score']),2)}] {doc.content}" for doc in documents])
    return ([Document(content=content)],)


REGISTERED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "rename": rename,
    # "value_to_list": value_to_list,
    # "join_lists": join_lists,
    # "join_strings": join_strings,
    # "join_documents": join_documents,
    "join_documents_and_scores": join_documents_and_scores,
    "strings_to_answers": strings_to_answers,
    # "answers_to_strings": answers_to_strings,
    # "strings_to_documents": strings_to_documents,
    # "documents_to_strings": documents_to_strings,
}


class Shaper(BaseComponent):

    """
    Shaper is a component that can invoke arbitrary, registered functions on the invocation context
    (query, documents, and so on) of a pipeline. It then passes the new or modified variables further down the pipeline.

    Using YAML configuration, the Shaper component is initialized with functions to invoke on pipeline invocation
    context.

    For example, in the YAML snippet below:
    ```yaml
        components:
        - name: shaper
          type: Shaper
          params:
            func: value_to_list
            inputs:
                value: query
                target_list: documents
            output: [questions]
    ```
    the Shaper component is initialized with a directive to invoke function expand on the variable query and to store
    the result in the invocation context variable questions. All other invocation context variables are passed down
    the pipeline as they are.

    You can use multiple Shaper components in a pipeline to modify the invocation context as needed.

    Currently, `Shaper` supports the following functions:

    - `rename`
    - `value_to_list`
    - `join_lists`
    - `join_strings`
    - `format_string`
    - `join_documents`
    - `join_documents_and_scores`
    - `format_document`
    - `format_answer`
    - `join_documents_to_string`
    - `strings_to_answers`
    - `string_to_answer`
    - `parse_references`
    - `answers_to_strings`
    - `join_lists`
    - `strings_to_documents`
    - `documents_to_strings`

    See their descriptions in the code for details about their inputs, outputs, and other parameters.
    """

    outgoing_edges = 1

    def __init__(
        self,
        func: str,
        outputs: List[str],
        inputs: Optional[Dict[str, Union[List[str], str]]] = None,
        params: Optional[Dict[str, Any]] = None,
        publish_outputs: Union[bool, List[str]] = True,
    ):
        """
        Initializes the Shaper component.

        Some examples:

        ```yaml
        - name: shaper
          type: Shaper
          params:
          func: value_to_list
          inputs:
            value: query
            target_list: documents
          outputs:
            - questions
        ```
        This node takes the content of `query` and creates a list that contains the value of `query` `len(documents)` times.
        This list is stored in the invocation context under the key `questions`.

        ```yaml
        - name: shaper
          type: Shaper
          params:
          func: join_documents
          inputs:
            value: documents
          params:
            delimiter: ' - '
          outputs:
            - documents
        ```
        This node overwrites the content of `documents` in the invocation context with a list containing a single Document
        whose content is the concatenation of all the original Documents. So if `documents` contained
        `[Document("A"), Document("B"), Document("C")]`, this shaper overwrites it with `[Document("A - B - C")]`

        ```yaml
        - name: shaper
          type: Shaper
          params:
          func: join_strings
          params:
            strings: ['a', 'b', 'c']
            delimiter: ' . '
          outputs:
            - single_string

        - name: shaper
          type: Shaper
          params:
          func: strings_to_documents
          inputs:
            strings: single_string
            metadata:
              name: 'my_file.txt'
          outputs:
            - single_document
        ```
        These two nodes, executed one after the other, first add a key in the invocation context called `single_string`
        that contains `a . b . c`, and then create another key called `single_document` that contains instead
        `[Document(content="a . b . c", metadata={'name': 'my_file.txt'})]`.

        :param func: The function to apply.
        :param inputs: Maps the function's input kwargs to the key-value pairs in the invocation context.
            For example, `value_to_list` expects the `value` and `target_list` parameters, so `inputs` might contain:
            `{'value': 'query', 'target_list': 'documents'}`. It doesn't need to contain all keyword args, see `params`.
        :param params: Maps the function's input kwargs to some fixed values. For example, `value_to_list` expects
            `value` and `target_list` parameters, so `params` might contain
            `{'value': 'A', 'target_list': [1, 1, 1, 1]}` and the node's output is `["A", "A", "A", "A"]`.
            It doesn't need to contain all keyword args, see `inputs`.
            You can use params to provide fallback values for arguments of `run` that you're not sure exist.
            So if you need `query` to exist, you can provide a fallback value in the params, which will be used only if `query`
            is not passed to this node by the pipeline.
        :param outputs: The key to store the outputs in the invocation context. The length of the outputs must match
            the number of outputs produced by the function invoked.
        :param publish_outputs: Controls whether to publish the outputs to the pipeline's output.
            Set `True` (default value) to publishes all outputs or `False` to publish None.
            E.g. if `outputs = ["documents"]` result for `publish_outputs = True` looks like
            ```python
                {
                    "invocation_context": {
                        "documents": [...]
                    },
                    "documents": [...]
                }
            ```
            For `publish_outputs = False` result looks like
            ```python
                {
                    "invocation_context": {
                        "documents": [...]
                    },
                }
            ```
            If you want to have finer-grained control, pass a list of the outputs you want to publish.
        """
        super().__init__()
        self.function = REGISTERED_FUNCTIONS[func]
        self.outputs = outputs
        self.inputs = inputs or {}
        self.params = params or {}
        if isinstance(publish_outputs, bool):
            self.publish_outputs = self.outputs if publish_outputs else []
        else:
            self.publish_outputs = publish_outputs

    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        invocation_context = invocation_context or {}
        if query and "query" not in invocation_context.keys():
            invocation_context["query"] = query

        if file_paths and "file_paths" not in invocation_context.keys():
            invocation_context["file_paths"] = file_paths

        if labels and "labels" not in invocation_context.keys():
            invocation_context["labels"] = labels

        if documents and "documents" not in invocation_context.keys():
            invocation_context["documents"] = documents

        if meta and "meta" not in invocation_context.keys():
            invocation_context["meta"] = meta

        input_values: Dict[str, Any] = {}
        for key, value in self.inputs.items():
            if isinstance(value, list):
                input_values[key] = []
                for v in value:
                    if v in invocation_context.keys() and v is not None:
                        input_values[key].append(invocation_context[v])
            else:
                if value in invocation_context.keys() and value is not None:
                    input_values[key] = invocation_context[value]

        # auto fill in input values if there's an invocation context value with the same name
        function_params = inspect.signature(self.function).parameters
        for parameter in function_params.values():
            if (
                parameter.name not in input_values.keys()
                and parameter.name not in self.params.keys()
                and parameter.name in invocation_context.keys()
            ):
                input_values[parameter.name] = invocation_context[parameter.name]

        input_values = {**self.params, **input_values}
        try:
            logger.debug(
                "Shaper is invoking this function: %s(%s)",
                self.function.__name__,
                ", ".join([f"{key}={value}" for key, value in input_values.items()]),
            )
            output_values = self.function(**input_values)
            if not isinstance(output_values, tuple):
                output_values = (output_values,)
        except TypeError as e:
            raise ValueError(
                "Shaper couldn't apply the function to your inputs and parameters. "
                "Check the above stacktrace and make sure you provided all the correct inputs, parameters, "
                "and parameter types."
            ) from e

        if len(self.outputs) < len(output_values):
            logger.warning(
                "The number of outputs from function %s is %s. However, only %s output key(s) were provided. "
                "Only %s output(s) will be stored. "
                "Provide %s output keys to store all outputs.",
                self.function.__name__,
                len(output_values),
                len(self.outputs),
                len(self.outputs),
                len(output_values),
            )

        if len(self.outputs) > len(output_values):
            logger.warning(
                "The number of outputs from function %s is %s. However, %s output key(s) were provided. "
                "Only the first %s output key(s) will be used.",
                self.function.__name__,
                len(output_values),
                len(self.outputs),
                len(output_values),
            )

        results = {}
        for output_key, output_value in zip(self.outputs, output_values):
            invocation_context[output_key] = output_value
            if output_key in self.publish_outputs:
                results[output_key] = output_value
        results["invocation_context"] = invocation_context

        return results, "output_1"

    def run_batch(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        return self.run(
            query=query,
            file_paths=file_paths,
            labels=labels,
            documents=documents,
            meta=meta,
            invocation_context=invocation_context,
        )


class BaseOutputParser(Shaper):
    """
    An output parser in `PromptTemplate` defines how to parse the model output and convert it into Haystack primitives (answers, documents, or labels).
    BaseOutputParser is the base class for output parser implementations.
    """

    @property
    def output_variable(self) -> Optional[str]:
        return self.outputs[0]


class AnswerParser(BaseOutputParser):
    """
    Parses the model output to extract the answer into a proper `Answer` object using regex patterns.
    AnswerParser adds the `document_ids` of the documents used to generate the answer and the prompts used to the `Answer` object.
    You can pass a `reference_pattern` to extract the document_ids of the answer from the model output.
    """

    def __init__(self, pattern: Optional[str] = None, reference_pattern: Optional[str] = None):
        """
         :param pattern: The regex pattern to use for parsing the answer.
            Examples:
                `[^\\n]+$` finds "this is an answer" in string "this is an argument.\nthis is an answer".
                `Answer: (.*)` finds "this is an answer" in string "this is an argument. Answer: this is an answer".
            If not specified, the whole string is used as the answer. If specified, the first group of the regex is used as the answer. If there is no group, the whole match is used as the answer.
        :param reference_pattern: The regex pattern to use for parsing the document references.
            Example: `\\[(\\d+)\\]` finds "1" in string "this is an answer[1]".
            If None, no parsing is done and all documents are referenced.
        """
        self.pattern = pattern
        self.reference_pattern = reference_pattern
        super().__init__(
            func="strings_to_answers",
            inputs={"strings": "results"},
            outputs=["answers"],
            params={"pattern": pattern, "reference_pattern": reference_pattern},
        )
