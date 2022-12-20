# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import List, Dict, Union, Optional, Type
from rich.console import Console
from rich.theme import Theme
from rich.markdown import Markdown
from rich.table import Table
from rich.highlighter import RegexHighlighter


def _get_highlighter(word: str) -> Type[RegexHighlighter]:
    """construct Regex Highlighter class based on the word

    Args:
        word (str): the query word

    Returns:
        Type[RegexHighlighter]: the sub-class of RegexHighlighter
    """

    class KeywordHighlighter(RegexHighlighter):
        base_style = "paddlenlp."
        highlights = [f"(?P<keyword>{word})"]

    return KeywordHighlighter()


def print_example_code():
    # 1. define the console
    console = Console()
    markdown = """
## you can download the above model with the following command:

### ***paddlenlp download --cache-dir ./paddle_pretrained_models <model name>***

### ***the <model name> is copied from above table***
    """
    console.print(Markdown(markdown))


def tabulate(
    tables: List[Union[List[str], Dict[str, str]]],
    headers: Optional[List[str]] = None,
    highlight_word: Optional[str] = None,
):
    """print tabulate data into console

    Args:
        tables (List[Union[List[str], Dict[str, str]]]): the table instance data
        headers (Optional[List[str]], optional): the header configuration. Defaults to None.
        highlight_word (Optional[str], optional): the highlight word. Defaults to None.
    """
    # 1. define the console
    theme = Theme({"paddlenlp.keyword": "bold magenta"})
    console = Console(highlighter=_get_highlighter(highlight_word), theme=theme)
    table_instance = Table(
        title="PaddleNLP 模型检索结果", show_header=headers is not None, header_style="bold magenta", highlight=True
    )

    # 2. add column
    headers = headers or []
    for header in headers:
        if isinstance(header, str):
            table_instance.add_column(header)
        else:
            table_instance.add_column(**header)

    # 3. add row data
    for row_data in tables:
        table_instance.add_row(*row_data)

    console.print(table_instance, justify="center")
