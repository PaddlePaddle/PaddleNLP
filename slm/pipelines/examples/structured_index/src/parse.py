# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Dict, List

from .utils import get_messages, get_response, load_data, load_model


class DocumentStructureParser:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-7B-Instruct",
        url: str = None,
    ):
        self.model_name = model_name_or_path
        if url is None:
            self.model, self.tokenizer = load_model(model_name_or_path=model_name_or_path)
        self.url = url
        self.prompt_template = """**Structured document text**
Insert # into the text, using the number of # to represent the hierarchical level of the content, while keeping all the original information.
The structured results are output in the form of markdown.
```markdown
```

**Input text:**
[input]
**Structured Text:**
"""

    def __call__(self, file_path: str) -> Dict:
        data = load_data(file_path=file_path, mode="Parsing")
        return self.parse_doc(data=data)

    def extract_text(self, response: str) -> str:
        """
        Extract text content from a markdown code block within a given response string.

        Args:
            response (str): The response string containing markdown code blocks.

        Returns:
            str: The extracted text content from the markdown code block.
        """
        start_index = response.rfind("```markdown") + len("```markdown")
        end_index = response.rfind("```")
        content = response[start_index:end_index]
        return content.strip()

    def parse_doc(self, data: Dict, max_text_length=512, repeat_length=2048) -> Dict:
        """
        Parse a document by extracting its content, formatting it according to a prompt template, and structuring the content into a hierarchical format.

        Args:
            data (Dict): A dictionary containing the document's data, including nodes with content.
            max_text_length (int, optional): The maximum length of text to consider from each node. Defaults to 512.
            repeat_length (int, optional): The length at which the text should be repeated and inserted into the prompt template. Defaults to 2048.

        Returns:
            Dict: A dictionary containing the parsed and structured content of the document.
        """
        contents = [item["content"] for item in data["nodes"]]
        content = "".join(c[:max_text_length].strip() for c in contents)
        content = self.prompt_template.replace("[input]", content)
        if len(content) <= repeat_length:
            content = content.replace("Insert # into the text,", "Repeat the text and insert # into the text,")

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        logging.debug(messages[0]["content"])
        if self.url is None:
            predicted_str = get_response(messages=messages, tokenizer=self.tokenizer, model=self.model)
        else:
            predicted_str = get_messages(
                messages=messages,
                model_name=self.model_name,
                url=self.url,
                temperature=0.0,
            )[0]
        logging.debug(f"parse result =\n{predicted_str}")
        predicted_str = self.extract_text(response=predicted_str)

        predicted_str_list: List[str] = predicted_str.splitlines()
        cur_dictionary = []
        node_id = -1
        data["nodes"] = []
        for line in predicted_str_list:
            line = line.strip()
            if len(line) == 0:
                continue
            node_id += 1
            level = -1
            while line.startswith("#"):
                level += 1
                line = line[1:]
            content = line.strip()
            if level == -1:
                level = len(cur_dictionary)
            data["nodes"].append({"id": node_id, "content": content, "level": level})

            while len(cur_dictionary) > level:
                cur_dictionary.pop()
            parent_id = cur_dictionary[-1] if len(cur_dictionary) > 0 else -1
            cur_dictionary.append(node_id)
            assert len(cur_dictionary) <= level + 1, f"len={len(cur_dictionary)}, level={level}"

            if parent_id > -1:
                data["nodes"][node_id]["parent_id"] = parent_id
                if "child_ids" not in data["nodes"][parent_id]:
                    data["nodes"][parent_id]["child_ids"] = []
                data["nodes"][parent_id]["child_ids"].append(node_id)

        return data
