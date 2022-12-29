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

import json


def json_format_indent(json_file, output_json):
    with open(output_json, "w", encoding="utf-8") as wf:
        with open(json_file, "r", encoding="utf-8") as rf:
            all_lines = []
            for json_line in rf:
                line_dict = json.loads(json_line)
                all_lines.append(line_dict)
            output_dataset = {"data": all_lines}
            json.dump(output_dataset, wf, ensure_ascii=False, indent="\t")


if __name__ == "__main__":
    json_format_indent("", "")
