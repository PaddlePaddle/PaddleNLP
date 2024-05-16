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

import json

file = "./dev.json"
target_file = "./result/dev.json"


def reconstruct_messages(messages: list[dict]):
    result = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            assistant_function = dict(
                role="assistant",
                function_call=dict(
                    thoughts=result[-1]["content"], name=message["name"], parameters=message["parameters"]
                ),
                content=message["observation"],
            )
            result[-1] = assistant_function
        else:
            result.append(message)
    return result


raws = []
with open(file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        messages = reconstruct_messages(data["conversations"])
        raws.append(dict(tools=data["tools"], messages=messages))

with open(target_file, "w+", encoding="utf-8") as f:
    import pdb

    pdb.set_trace()
    for raw in raws:
        data = json.dumps(raw, ensure_ascii=False)
        f.write(data + "\n")
