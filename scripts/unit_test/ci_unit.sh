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
export log_path=/workspace/PaddleNLP/unittest_logs
mkdir -p /workspace/PaddleNLP/coverage_report
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
    python -m pip install allure-pytest -y
    python -m pip install --no-cache-dir ${paddle}
    python -c "import paddle;print('paddle');print(paddle.__version__);print(paddle.version.show())" >> ${log_path}/commit_info.txt

    python setup.py bdist_wheel > /dev/null
    python -m pip install  dist/p****.whl
    python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)" >> ${log_path}/commit_info.txt
    
    if git diff --numstat "$AGILE_COMPILE_BRANCH" | awk '{print $NF}' | grep -q '^csrc/'; then
        echo "Found modifications in csrc, running setup_cuda.py install and uploading it to bos."
        cd ${nlp_dir}/csrc
        # python setup_cuda.py install
        bash tools/build_wheel.sh python3.10 80
        # cp ./dist/p****.whl ${PPNLP_HOME}/upload/
        # cd ${PPNLP_HOME}
        # python upload.py ${PPNLP_HOME}/upload 'paddlenlp/wheels'
        # rm -rf upload/*
    else
        echo "No modifications in csrc, installing paddlenlp_ops wheel file..."
        python -m pip install https://paddlenlp.bj.bcebos.com/wheels/paddlenlp_ops-0.0.0-py3-none-any.whl
    fi

    pip list 
}

set_env() {
    export NVIDIA_TF32_OVERRIDE=0 
    export FLAGS_cudnn_deterministic=1
    export HF_ENDPOINT=https://hf-mirror.com
    export FLAGS_use_cuda_managed_memory=true
}

print_info() {
    if [ $1 -ne 0 ]; then
        cat ${log_path}/unittest.log | grep -v "Fail to fscanf: Success" \
            | grep -v "SKIPPED" | grep -v "warning" > ${log_path}/unittest_FAIL.log
        tail -n 1 ${log_path}/unittest.log >> ${log_path}/unittest_FAIL.log
        echo -e "\033[31m ${log_path}/unittest_FAIL \033[0m"
        cat ${log_path}/unittest_FAIL.log
        cp ${log_path}/unittest_FAIL.log ${PPNLP_HOME}/upload/unittest_FAIL.log.${AGILE_PIPELINE_BUILD_ID}.${AGILE_JOB_BUILD_ID}
        cd ${PPNLP_HOME} && python upload.py ${PPNLP_HOME}/upload 'paddlenlp/PaddleNLP_CI/PaddleNLP-CI-Unittest-GPU'
        rm -rf upload/*
    else
        tail -n 1 ${log_path}/unittest.log
        echo -e "\033[32m ${log_path}/unittest_SUCCESS \033[0m"
    fi
}

install_requirements
set_env
cd ${nlp_dir}
echo ' Testing all unittest cases '
pytest -v -n 8 \
  --dist loadgroup \
  --retries 1 --retry-delay 1 \
  --timeout 200 --durations 20 --alluredir=result \
  --cov paddlenlp --cov-report xml:coverage.xml > ${log_path}/unittest.log 2>&1
exit_code=$?
print_info $exit_code unittest

cd ${nlp_dir}
echo -e "\033[35m ---- Genrate Allure Report  \033[0m"
unset http_proxy && unset https_proxy
cp scripts/regression/gen_allure_report.py ./
python gen_allure_report.py > ${nlp_dir}/coverage_report/gen_allure_report.log 2>&1
echo -e "\033[35m ---- Report: https://xly.bce.baidu.com/ipipe/ipipe-report/report/${AGILE_JOB_BUILD_ID}/report/  \033[0m"

exit $exit_code