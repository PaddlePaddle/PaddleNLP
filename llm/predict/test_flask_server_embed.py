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

import time

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community import vectorstores
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

all_items = "hello world \t hello world hello world\n hello world"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=20, separators=["\t", "\n", "。", "\n\n", ""]
)
texts = text_splitter.split_text("\t".join(all_items))

all_splits = [Document(page_content=text) for text in texts]
# 着重关注OpenAIEmbeddings类的调用过程，该类涉及客户端调用服务端接口过程，详细可查看相关源码实现，最后会定位到一次response接口的调用，只需保持response结果与标准OpenAI接口返回结果（即前文embedding接口数据返回格式）一致即可
embedding = OpenAIEmbeddings(
    base_url="http://127.0.0.1:8000/v1/", api_key="EMPTY", model="intfloat/e5-base-v2", tiktoken_enabled=False
)
vector = FAISS.from_documents(
    documents=all_splits,
    embedding=embedding,
)
vector = vector.serialize_to_bytes()

vectorstore = vectorstores.FAISS.deserialize_from_bytes(
    vector,
    embedding,
    allow_dangerous_deserialization=True,
)

key_list = ["hello,world"]
Q = []
C = []
for key in key_list:
    QUESTION = f"抽取关键信息:{key}"
    # c_str = ""
    Q.append(QUESTION)
    time.sleep(0.5)
    docs = vectorstore.similarity_search_with_relevance_scores(QUESTION, k=2)
    context = [(document.page_content, score) for document, score in docs]
    context = sorted(context, key=lambda x: x[1])
    C.extend([x[0] for x in context[::-1]])

C = list(set(C))
all_C = " ".join(C)
print(all_C)
