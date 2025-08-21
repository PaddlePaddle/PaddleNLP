#!/usr/bin/env bash

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

set -e
export nlp_dir=/workspace/PaddleNLP
mkdir -p /workspace/PaddleNLP/build_logs
export log_path=/workspace/PaddleNLP/build_logs
mkdir -p ${PPNLP_HOME}/upload_${AGILE_PIPELINE_BUILD_NUMBER}
upload_path=${PPNLP_HOME}/upload_${AGILE_PIPELINE_BUILD_NUMBER}
export Build_list=()

python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --user set global.trusted-host pypi.tuna.tsinghua.edu.cn

get_diff_case(){
    git diff --name-only HEAD~1 HEAD
    for file_name in `git diff --name-only HEAD~1 HEAD`;do
        arr_file_name=(${file_name//// })
        if [[ ${arr_file_name[0]} == "paddlenlp" ]];then
            Build_list[${#Build_list[*]}]="paddlenlp"
        elif [[ ${arr_file_name[0]} == "requirements"* ]]; then
            Build_list[${#Build_list[*]}]="paddlenlp"
        elif [[ ${arr_file_name[0]} == "csrc" ]];then
            Build_list[${#Build_list[*]}]="paddlenlp_ops"
        else
            continue
        fi
    done
    echo ${Build_list[*]}
}

install_paddle(){
    echo -e "\033[35m ---- Install paddlepaddle-gpu  \033[0m"
    python -m pip uninstall paddlepaddle -y
    python -m pip install pillow
    python -m pip install --user ${paddle} --no-cache-dir;
    python -c "import paddle;print('paddle');print(paddle.__version__); \
        print(paddle.version.show())" >> ${log_path}/commit_info.txt
}

paddlenlp_build (){
    echo -e "\033[32m ---- make PaddleNLP.tar.gz  \033[0m"
    cd /workspace
    tar -zcf PaddleNLP.tar.gz PaddleNLP/
    mv PaddleNLP.tar.gz ${upload_path}/

    echo -e "\033[35m ---- build latest paddlenlp  \033[0m"
    cd $nlp_dir
    rm -rf build/
    rm -rf paddlenlp.egg-info/
    rm -rf ppdiffusers.egg-info/
    rm -rf paddle_pipelines.egg-info/
    rm -rf dist/

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python setup.py bdist_wheel
    python -m pip install --ignore-installed  dist/p****.whl --force-reinstall
    python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)" >> ${log_path}/commit_info.txt

    # for test https://www.paddlepaddle.org.cn/whl/paddlenlp.html
    cp $nlp_dir/dist/p****.whl ${upload_path}/
    # for ci pr test
    cp $nlp_dir/dist/p****.whl ${upload_path}/paddlenlp-ci-py3-none-any.whl
}

install_paddlenlp(){
    echo "install_nlp_develop"
    python -m pip uninstall protobuf -y
    python -m pip install protobuf==3.20.2
    python -m pip install numpy==1.26.4 --force-reinstall
    python -m pip install --user https://paddlenlp.bj.bcebos.com/wheels/paddlenlp-ci-py3-none-any.whl --no-cache-dir
    python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)" >> ${log_path}/commit_info.txt
}

paddlenlp_ops_build (){
    cd ${nlp_dir}/csrc
    bash tools/build_wheel.sh
    python -c "import paddlenlp_ops"
    # for test https://www.paddlepaddle.org.cn/whl/paddlenlp.html
    cp ${nlp_dir}/csrc/gpu_dist/p****.whl ${upload_path}/
    # for ci pr test
    cp ${nlp_dir}/csrc/gpu_dist/p****.whl ${upload_path}/paddlenlp_ops-ci-py3-none-any.whl
}

contain_case(){
    local e
    for e in "${@:2}";do
        if [[ "$e" == "$1" ]];then
            return 1
        fi
    done
    return 0
}

### main
cd ${nlp_dir}
get_diff_case
Build_list=($(awk -v RS=' ' '!a[$1]++' <<< ${Build_list[*]}))
if [[ ${#Build_list[*]} -ne 0 ]];then
    echo -e "\033[31m ---- Build_list length: ${#Build_list[*]}, cases: ${Build_list[*]} \033[0m"
    echo -e "\033[31m ============================= \033[0m"
    install_paddle
    if [[ $(contain_case paddlenlp ${Build_list[@]}; echo $?) -eq 1 ]];then
        paddlenlp_build
    else
        install_paddlenlp
    fi
    
    if [[ $(contain_case paddlenlp_ops ${Build_list[@]}; echo $?) -eq 1 ]];then
        paddlenlp_ops_build
    fi

    if [ -e "${upload_path}" ] && [ "$(ls -A "${upload_path}/")" ]; then
        cd ${upload_path} && ls -A "${upload_path}"
        cd ${PPNLP_HOME} && python upload.py ${upload_path} 'paddlenlp/wheels'
        rm -rf ${upload_path}
        echo -e "\033[32m upload wheels SUCCESS \033[0m"
    fi
else
    echo -e "\033[32m Don't need build any whl  \033[0m"
fi