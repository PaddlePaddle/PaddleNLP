# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re

from jinja2 import Template

MODEL_ROOT = "/xx/bos/community/"
URL_BASE = "https://paddlenlp.bj.bcebos.com/models/community/"
OUTPUT_DIR = "./website"

MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Downloads</title>
</head>
<body>
    <h1>Available Models</h1>
    <ul>
        {% for model in models %}
        <li><a href="{{ model }}/index.html">{{ model }}</a></li>
        {% endfor %}
    </ul>
</body>
</html>
"""

MODEL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ model_name }}</title>
</head>
<body>
    <h1>{{ model_name }} Files</h1>
    <ul>
        {% for file in files %}
        <li>
            <a href="{{ model_path }}/{{ file.name }}" download>{{ file.name }}</a>
            <span>({{ file.size }})</span>
        </li>
        {% endfor %}
    </ul>
    <p><a href="../../">Back to Main</a></p>
</body>
</html>
"""


def convert_size(size_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024.0
        unit_index += 1
    return f"{size_bytes:.1f} {units[unit_index]}"


def generate_model_page(model_path, model_name):
    full_path = os.path.join(MODEL_ROOT, model_path)
    files = []

    for root, _, filenames in os.walk(full_path):
        for f in filenames:
            if f.endswith("index.html"):
                continue
            file_path = os.path.join(root, f)
            rel_path = os.path.relpath(file_path, full_path)

            size = os.path.getsize(file_path)
            files.append({"name": rel_path, "size": convert_size(size)})

    output_path = os.path.join(OUTPUT_DIR, model_path)
    os.makedirs(output_path, exist_ok=True)

    template = Template(MODEL_TEMPLATE)
    html = template.render(
        model_name=model_name, model_path=URL_BASE + model_path, files=sorted(files, key=lambda x: x["name"])
    )

    with open(os.path.join(output_path, "index.html"), "w") as f:
        f.write(html)


def generate_main_page(models):
    template = Template(MAIN_TEMPLATE)
    html = template.render(models=sorted(models))

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
        f.write(html)


def is_model_directory(path):
    if os.path.isfile(os.path.join(path, "model_index.json")):
        return True

    if not os.path.isfile(os.path.join(path, "config.json")):
        return False

    model_files = [
        f
        for f in os.listdir(path)
        if f.startswith(("model", "pytorch_model"))
        and (f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pdparams"))
    ]
    sharded_files = [f for f in os.listdir(path) if re.match(r"model-\d+-of-\d+\.safetensors", f)]
    return len(model_files) > 0 or len(sharded_files) > 0


ommit_paths = ["_internal_", "hf-internal", "zhuweiguo"]


def find_models():
    models = []
    for root, dirs, _ in os.walk(MODEL_ROOT):
        rel_path = os.path.relpath(root, MODEL_ROOT)
        if any(p in rel_path for p in ommit_paths):
            continue
        print(rel_path)
        if rel_path == ".":
            continue

        if is_model_directory(root):
            models.append(rel_path)
            dirs[:] = []
    return models


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = find_models()
    generate_main_page(models)

    for model_path in models:
        model_name = os.path.basename(model_path)
        generate_model_page(model_path, model_name)


if __name__ == "__main__":
    main()
