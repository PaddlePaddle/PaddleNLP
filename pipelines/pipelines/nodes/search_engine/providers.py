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

import json
import logging
from typing import Any, Dict, List, Optional, Union

import requests

from pipelines import Document
from pipelines.nodes.search_engine.base import SearchEngine

logger = logging.getLogger(__name__)


class SerpAPI(SearchEngine):
    """
    SerpAPI is a search engine that provides a REST API to access search results from Google, Bing, Yahoo, Yandex,
    Amazon, and similar. See the [SerpAPI website](https://serpapi.com/) for more details.
    """

    def __init__(
        self,
        api_key: str,
        top_k: Optional[int] = 10,
        engine: Optional[str] = "google",
        search_engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_key: API key for SerpAPI.
        :param top_k: Number of results to return.
        :param engine: Search engine to use, for example google, bing, baidu, duckduckgo, yahoo, yandex.
        See the [SerpAPI documentation](https://serpapi.com/search-api) for the full list of supported engines.
        :param search_engine_kwargs: Additional parameters passed to the SerperDev API. For example, you can set 'lr' to 'lang_en'
        to limit the search to English.
        See the [SerpAPI documentation](https://serpapi.com/search-api) for the full list of supported parameters.
        """
        super().__init__()
        self.params_dict: Dict[str, Union[str, int, float]] = {}
        self.api_key = api_key
        self.kwargs = search_engine_kwargs if search_engine_kwargs else {}
        self.engine = engine
        self.top_k = top_k

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the SerpAPI. For example, you can set 'lr' to 'lang_en'
        to limit the search to English.
        See the [SerpAPI documentation](https://serpapi.com/search-api) for the full list of supported parameters.
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}
        top_k = kwargs.pop("top_k", self.top_k)
        url = "https://serpapi.com/search"

        params = {"source": "python", "serp_api_key": self.api_key, "q": query, **kwargs}

        if self.engine:
            params["engine"] = self.engine
        response = requests.get(url, params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = json.loads(response.text)
        organic = [
            Document.from_dict(d, field_map={"snippet": "content"})
            for d in json_result["organic_results"]
            if "snippet" in d
        ]
        answer_box = []
        if "answer_box" in json_result:
            answer_dict = json_result["answer_box"]
            for key in ["answer", "snippet_highlighted_words", "snippet", "title"]:
                if key in answer_dict:
                    answer_box_content = answer_dict[key]
                    if isinstance(answer_box_content, list):
                        answer_box_content = answer_box_content[0]
                    answer_box = [
                        Document.from_dict(
                            {
                                "title": answer_dict.get("title", ""),
                                "content": answer_box_content,
                                "link": answer_dict.get("displayed_link", ""),
                            }
                        )
                    ]
                    break

        people_also_search = []
        if "people_also_search_for" in json_result:
            for result in json_result["people_also_search_for"]:
                people_also_search.append(
                    Document.from_dict(
                        {
                            "title": result["title"],
                            "content": result["snippet"] if result.get("snippet") else result["title"],
                            "link": result["link"],
                        }
                    )
                )

        related_questions = []
        if "related_questions" in json_result:
            for result in json_result["related_questions"]:
                related_questions.append(
                    Document.from_dict(
                        {
                            "title": result["title"],
                            "content": result["snippet"] if result.get("snippet") else result["title"],
                            "link": result["link"],
                        }
                    )
                )

        documents = answer_box + organic + people_also_search + related_questions

        logger.debug("SerpAPI returned %s documents for the query '%s'", len(documents), query)
        result_docs = documents[:top_k]
        return self.score_results(result_docs, len(answer_box) > 0)


class SerperDev(SearchEngine):
    """
    Serper.dev is a search engine that provides a REST API to access search results from Google. See the [Serper.dev website](https://serper.dev.com/) for more details.
    """

    def __init__(
        self,
        api_key: str,
        top_k: Optional[int] = 10,
        engine: Optional[str] = "google",
        search_engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_key: API key for Serper.dev API.
        :param top_k: Number of results to return.
        :param engine: Search engine to use, only supports Google.
        :param search_engine_kwargs: Additional parameters passed to the SerperDev API. For example, you can set 'hl' to 'en'
        to set the search results language to English.
        See the [Serper.dev documentation](https://serper.dev/playground) for the full list of supported parameters.
        """
        super().__init__()
        self.params_dict: Dict[str, Union[str, int, float]] = {}
        self.api_key = api_key
        self.kwargs = search_engine_kwargs if search_engine_kwargs else {}
        self.engine = engine
        self.top_k = top_k

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        :param query: Query string.
        :param kwargs: Additional parameters passed to the Serper.dev API. For example, you can set 'hl' to 'en'
        to set the search results language to English.
        See the [Serper.dev documentation](https://serper.dev/playground) for the full list of supported parameters.
        :return: List[Document]
        """
        kwargs = {**self.kwargs, **kwargs}
        top_k = kwargs.pop("top_k", self.top_k)
        url = "https://google.serper.dev/search"

        params = {"q": query, **kwargs}

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Error while querying {self.__class__.__name__}: {response.text}")

        json_result = json.loads(response.text)
        organic = [
            Document.from_dict(d, field_map={"snippet": "content"}) for d in json_result["organic"] if "snippet" in d
        ]
        answer_box = []
        if "answerBox" in json_result:
            answer_dict = json_result["answerBox"]
            for key in ["answer", "snippetHighlighted", "snippet", "title"]:
                if key in answer_dict:
                    answer_box_content = answer_dict[key]
                    if isinstance(answer_box_content, list):
                        answer_box_content = answer_box_content[0]
                    answer_box = [
                        Document.from_dict(
                            {
                                "title": answer_dict.get("title", ""),
                                "content": answer_box_content,
                                "link": answer_dict.get("link", ""),
                            }
                        )
                    ]
                    break

        people_also_search = []
        if "peopleAlsoSearchFor" in json_result:
            for result in json_result["peopleAlsoSearchFor"]:
                people_also_search.append(
                    Document.from_dict(
                        {
                            "title": result["title"],
                            "content": result["snippet"] if result.get("snippet") else result["title"],
                            "link": result["link"],
                        }
                    )
                )

        related_searches = []
        if "relatedSearches" in json_result:
            for result in json_result["relatedSearches"]:
                related_searches.append(Document.from_dict({"content": result.get("query", "")}))

        related_questions = []
        if "peopleAlsoAsk" in json_result:
            for result in json_result["peopleAlsoAsk"]:
                related_questions.append(
                    Document.from_dict(
                        {
                            "title": result["title"],
                            "content": result["snippet"] if result.get("snippet") else result["title"],
                            "link": result["link"],
                        }
                    )
                )

        documents = answer_box + organic + people_also_search + related_searches + related_questions

        logger.debug("Serper.dev API returned %s documents for the query '%s'", len(documents), query)
        result_docs = documents[:top_k]
        return self.score_results(result_docs, len(answer_box) > 0)
