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

import os

from setuptools import Distribution, setup

packages = []
package_data = {}


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def main():
    setup(
        name="paddlenlp_ops",
        version="0.0.0",
        description="PaddleNLP SDAA CustomOps",
        long_description="",
        long_description_content_type="text/markdown",
        author_email="Paddle-better@baidu.com",
        maintainer="PaddlePaddle",
        maintainer_email="Paddle-better@baidu.com",
        project_urls={},
        license="Apache Software License",
        packages=[
            "paddlenlp_ops",
        ],
        include_package_data=True,
        package_data={
            "": ["*.py"],
        },
        package_dir={
            "": "python",
        },
        zip_safe=False,
        distclass=BinaryDistribution,
        entry_points={"console_scripts": []},
        classifiers=[],
        keywords="PaddleNLP SDAA CustomOps",
    )


if __name__ == "__main__":
    main()
