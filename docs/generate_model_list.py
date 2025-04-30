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

# Markdown templates
MAIN_TEMPLATE = """
# Model Downloads

## Available Models

{% for model in models %}
- [{{ model }}]({{ model }}/index.md)
{% endfor %}
"""

MODEL_TEMPLATE = """
# {{ model_name }}
---

{% if readme_content %}
## README([From Huggingface]({{ huggingface_url }}))

{{ readme_content }}

{% endif %}

## Model Files
{% for file in files %}
- [{{ file.name }}]({{ model_path }}/{{ file.name }}) ({{ file.size }})
{% endfor %}

[Back to Main]({{back_to_main_path}})
"""


def convert_size(size_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024.0
        unit_index += 1
    return f"{size_bytes:.1f} {units[unit_index]}"


def process_image_links(text, model_path):
    image_link_pattern = re.compile(r"!\[.*?\]\((.*?)\)")

    image_links = image_link_pattern.findall(text)
    for i, link in enumerate(image_links):
        if not link.startswith(("http://", "https://", "/")):
            prefix = f"https://huggingface.co/{model_path}/resolve/main/"
            image_links[i] = prefix + link

    def replace_link(match):
        original_link = match.group(1)
        new_link = next((new_link for new_link in image_links if original_link in new_link), original_link)
        return f'![{match.group(0).split("](")[0]}]({new_link})'

    processed_text = image_link_pattern.sub(replace_link, text)
    return processed_text


def process_license(text):
    license_pattern = re.compile(r"---\nlicense:.*?---", re.DOTALL)
    processed_text = license_pattern.sub("", text)
    return processed_text


def get_back_to_main_path(model_path):
    # calculate the level by counting the number of slashes
    level = model_path.count("/") + 1

    # back_to_main_path = '../' * (level - 1)
    back_to_main_path = "../" * level
    return back_to_main_path


def generate_model_page(model_path, model_name):
    full_path = os.path.join(MODEL_ROOT, model_path)
    files = []
    readme_content = False

    for root, _, filenames in os.walk(full_path):
        for f in filenames:
            if f.endswith("index.md"):
                continue
            file_path = os.path.join(root, f)
            rel_path = os.path.relpath(file_path, full_path)

            if f == "README.md":
                with open(file_path, "r", encoding="utf-8") as rf:
                    readme_content = rf.read()

            size = os.path.getsize(file_path)
            files.append({"name": rel_path, "size": convert_size(size)})

    output_path = os.path.join(OUTPUT_DIR, model_path)
    os.makedirs(output_path, exist_ok=True)

    if readme_content:
        readme_content = process_image_links(readme_content, model_path)
        readme_content = process_license(readme_content)
    huggingface_url = os.path.join("https://huggingface.co", model_path)
    back_to_main_path = get_back_to_main_path(model_path)
    template = Template(MODEL_TEMPLATE)
    markdown_content = template.render(
        model_name=model_name,
        huggingface_url=huggingface_url,
        model_path=URL_BASE + model_path,
        files=sorted(files, key=lambda x: x["name"]),
        readme_content=readme_content,
        back_to_main_path=back_to_main_path,
    )

    with open(os.path.join(output_path, "index.md"), "w") as f:
        f.write(markdown_content)


def generate_main_page(models):
    template = Template(MAIN_TEMPLATE)
    markdown_content = template.render(models=sorted(models))

    with open(os.path.join(OUTPUT_DIR, "index.md"), "w") as f:
        f.write(markdown_content)


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


ommit_paths = [
    "_internal_",
    "hf-internal",
    "zhuweiguo",
    "ziqingyang",
    "yuhuili",
    "westfish",
    "junnyu",
    "Yang-Changhui",
    "baicai",
]


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
