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

from tqdm import tqdm

from .utils import get_messages, get_response, load_data, load_model


class Summarizer:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-7B-Instruct",
        url: str = None,
    ):
        self.model_name = model_name_or_path
        if url is None:
            self.model, self.tokenizer = load_model(model_name_or_path=model_name_or_path)
        self.url = url
        self.prompt_template = """# Generate text summaries based on text content and sub node summaries
**Task Description:**
You will receive a text and a summary of multiple child nodes that the text may depend on. Your task is to generate a concise and accurate summary for this text, no more than three sentences.
**Tips:**
1. Understand the text and child nodes: Carefully read the text itself and the summary of the child nodes (if any), and understand the logical relationship between them. Generate a summary with a focus on the text itself, noting that there may be missing spaces between words in the English text.
2. Integrate information: Integrate key information from the child node summary (if any) into the text summary to ensure the completeness and accuracy of the information.
3. Concise and clear: Use concise language to avoid redundant information and highlight key points. No more than two sentences.
4. Language consistency: The language used in the abstract should be consistent with the language of the text itself.
5. Compliant format: Output format is <summary>...</summary>， If unable to generate a summary, simply output an empty summary <summary></summary>.
**Input:**
<text>
Artificial intelligence (AI) is changing our lives. From autonomous vehicle to smart homes, AI has been used more and more widely.
</text>
<child_summary>
Autonomous vehicle use AI technology to achieve driverless driving and improve road safety and traffic efficiency.
The smart home system achieves automation control through AI technology, improving the convenience and comfort of life.
</child_summary>
**Output:**
<summary>
AI is changing our lives. Its applications include autonomous vehicle that improve road safety and traffic efficiency, and smart home systems that improve the convenience and comfort of life.
</summary>
**Input:**
<text>
[text]
</text>
<child_summary>
[child_summary]
</child_summary>
**Output:**"""

    def __call__(self, file_path: str) -> Dict:
        data = load_data(file_path=file_path, mode="Summarizing")
        return self.summarize_doc(data=data)

    def extract_summary(self, response: str) -> str:
        """
        Extract the summary content from a given response string enclosed within <summary> tags.

        Args:
            response (str): The response string containing the summary within <summary> tags.

        Returns:
            str: The extracted summary content.
        """
        start_index = response.rfind("<summary>") + len("<summary>")
        end_index = response.rfind("</summary>")
        content = response[start_index:end_index]
        return content.strip()

    def calc_topologic(self, nodes: List[Dict]) -> List[int]:
        """
        Calculate the topological order of nodes based on their parent-child relationships.

        Args:
            nodes (List[Dict]): A list of dictionaries representing nodes, each containing an 'id' and potentially a 'parent_id'.

        Returns:
            List[int]: A list of node IDs in topological order.
        """
        topologic = []
        pred = [0 for _ in range(len(nodes))]
        for node in nodes:
            if "parent_id" in node and node["parent_id"] >= 0:
                pred[node["parent_id"]] += 1
        while len(topologic) < len(nodes):
            for i in range(len(pred)):
                if pred[i] == 0:
                    topologic.append(i)
                    pred[i] = -1
                    if "parent_id" in nodes[i]:
                        pred[nodes[i]["parent_id"]] -= 1
        return topologic

    def summarize_doc(self, data: Dict) -> Dict:
        """
        Summarize a document by generating summaries for each node based on their content and child summaries.

        Args:
            data (Dict): A dictionary containing the document's data, including nodes with content and child relationships.

        Returns:
            Dict: A dictionary containing the summarized content for each node.
        """
        topologic = self.calc_topologic(nodes=data["nodes"])  # Calculate topological order
        logging.debug(f"Topologic: {topologic}")
        assert len(topologic) == len(data["nodes"])
        for cur_id in tqdm(topologic):
            cur_node = data["nodes"][cur_id]
            assert "summary" not in cur_node
            assert cur_id == cur_node["id"]

            message_content = self.prompt_template
            message_content = message_content.replace("[text]", cur_node["content"])
            child_summary = ""
            if "child_ids" in cur_node:
                for child_id in cur_node["child_ids"]:
                    assert "summary" in data["nodes"][child_id]
                    child_summary += data["nodes"][child_id]["summary"] + "\n"
            message_content = message_content.replace("[child_summary]", child_summary.strip())
            # logging.debug(f"Message content: {message_content}")
            messages = [
                {
                    "role": "user",
                    "content": message_content,
                }
            ]
            if self.url is None:
                response = get_response(messages=messages, tokenizer=self.tokenizer, model=self.model)
            else:
                response = get_messages(
                    messages=messages,
                    model_name=self.model_name,
                    url=self.url,
                    temperature=0.0,
                )[0]
            data["nodes"][cur_id]["summary"] = self.extract_summary(response=response)
            logging.debug(f"Summary for node {cur_id}: {data['nodes'][cur_id]['summary']}\n")

        return data
