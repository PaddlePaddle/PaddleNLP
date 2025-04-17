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
# flake8: noqa
from pipelines.utils.preprocessing import (  # isort: skip
    convert_files_to_dicts,
    convert_files_to_dicts_splitter,
    tika_convert_files_to_dicts,
)
from pipelines.utils.cleaning import clean_wiki_text
from pipelines.utils.doc_store import (
    launch_es,
    launch_milvus,
    launch_opensearch,
    launch_weaviate,
    stop_opensearch,
    stop_service,
)
from pipelines.utils.export_utils import (
    convert_labels_to_squad,
    export_answers_to_csv,
    print_answers,
    print_documents,
    print_questions,
)
from pipelines.utils.import_utils import fetch_archive_from_http
