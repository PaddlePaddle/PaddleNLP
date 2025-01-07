#!/usr/bin/env bash

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

export paddle=$1
export nlp_dir=/workspace/PaddleNLP
cd $nlp_dir

if [ ! -d "unittest_logs" ];then
    mkdir unittest_logs
fi

install_requirements() {
    python -m pip config --user set global.index http://pip.baidu-int.com/search/
    python -m pip config --user set global.index-url http://pip.baidu-int.com/simple
    python -m pip config --user set global.trusted-host pip.baidu-int.com
    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python -m pip install -r tests/requirements.txt
    python -m pip install -r paddlenlp/experimental/autonlp/requirements.txt 
    python -m pip uninstall paddlepaddle paddlepaddle_gpu -y
    python -m pip install pillow -y
    python -m pip install --no-cache-dir ${paddle}
    python -c "import paddle;print('paddle')print(paddle.__version__);print(paddle.version.show())" >> ${log_path}/commit_info.txt

    python setup.py bdist_wheel > /dev/null
    python -m pip install  dist/p****.whl
    python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)" >> ${log_path}/commit_info.txt
    cd csrc/
    python setup_cuda.py install
    cd ../

    pip list 
}

set_env() {
    export NVIDIA_TF32_OVERRIDE=0 
    export FLAGS_cudnn_deterministic=1
    export HF_ENDPOINT=https://hf-mirror.com
    export FLAGS_use_cuda_managed_memory=true
}

install_requirements
set_env
pytest -v -n 8 \
  --dist loadgroup \
  --retries 1 --retry-delay 1 \
  --timeout 200 --durations 20 \
  --cov paddlenlp --cov-report xml:coverage.xml
