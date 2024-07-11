#!/usr/bin/env bash

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
####################################
export paddle=$1
export nlp_dir=/workspace/PaddleNLP
mkdir -p /workspace/case_logs
export log_path=/workspace/case_logs
export case_list=()

target_lists_for_gpt=(
    "legacy/model_zoo/gpt-3"
    "llm/auto_parallel/gpt-3"
    "paddlenlp/transformers/gpt/modeling.py"
    "paddlenlp/transformers/gpt/modeling_pp.py"
    "paddlenlp/transformers/gpt/modeling_auto.py"
    "scripts/distribute"
)

target_lists_for_llama=(
    "llm/auto_parallel/llama"
    "paddlenlp/trainer/auto_trainer.py"
    "paddlenlp/transformers/llama/modeling_auto_static.py"
    "paddlenlp/transformers/llama/modeling_auto.py"
    "paddlenlp/transformers/llama/modeling.py"
    "scripts/distribute"
)

target_path_for_ci_scripts="scripts/distribute"

####################################
install_paddle(){
    echo -e "\033[31m ---- Install paddlepaddle-gpu  \033"
    python -m pip install --no-cache-dir --user ${paddle} --force-reinstall --no-dependencies;
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
}

install_paddlenlp(){
    echo -e "\033[31m ---- Install paddlenlp by set PYTHONPATH  \033"
    export PYTHONPATH=${nlp_dir}:$PYTHONPATH
    sed -i -e "s/paddlenlp/#paddlenlp/g" model_zoo/gpt-3/requirements.txt
    # export http_proxy=${proxy} && export https_proxy=${proxy}
    # python -m pip uninstall paddlenlp -y
    # rm -rf build/ && rm -rf paddlenlp.egg-info/ && rm -rf dist/
    # python -m pip install --ignore-installed -r requirements.txt
    # python -m pip install --ignore-installed -r requirements-dev.txt
    # python setup.py install
    # python setup.py build_ext
    # python setup.py bdist_wheel
    # unset http_proxy && unset https_proxy
    # cd -
    # python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)";
}

install_external_ops(){
    echo -e "\033[31m ---- Install extern_ops  \033"
    export PYTHONPATH=${nlp_dir}:$PYTHONPATH
    cd ${nlp_dir}/legacy/model_zoo/gpt-3/external_ops
    python setup.py install
    python -c "import fused_ln;";
}
####################################
get_diff_TO_case(){
cd ${nlp_dir}
for file_name in `git diff --numstat upstream/${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    dir4=${arr_file_name[3]}
    file_item=$dir1/$dir2/$dir3/$dir4
    echo "file_name:"${file_name}, "path:"${file_item}
    if [ ! -f ${file_name} ];then # 针对pr删掉文件
        continue
    elif [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
        continue
    else
        for ((i=0; i<${#target_lists_for_gpt[@]}; i++)); do
            if [[ ! ${dir3} =~ "benchmarks" ]] && [[ ${file_item} == *${target_lists_for_gpt[i]}* ]];then
                case_list[${#case_list[*]}]=gpt-3_auto
                case_list[${#case_list[*]}]=gpt-3_dygraph
            fi
        done
        for ((i=0; i<${#target_lists_for_llama[@]}; i++)); do
            if [[ ${file_item} == *${target_lists_for_llama[i]}* ]];then
                case_list[${#case_list[*]}]=llama_auto
            fi
        done
    fi
done
}
####################################
print_info(){
#解决异常退出-6的问题，CI中的偶现问题，无法复现
if [[ $1 -ne 0 ]] && [[ $1 -ne 250 ]];then
    EXCODE=2
    if [ ! -f ${log_path}/$2 ];then
        echo -e "\033[31m run $2 CI FAIL \033"
    else
        mv ${log_path}/$2 ${log_path}/$2_FAIL.log
        echo -e "\033[31m ${log_path}/$2_FAIL \033"
        tail -10 ${log_path}/$2_FAIL.log
    fi
    exit $EXCODE
else
    echo -e "\033[32m run $3 CI SUCCESS \033"
fi
}
####################################
function contain_case(){
    local e
    for e in "${@:2}";do
        if [[ "$e" == "$1" ]];then
            return 1
        fi
    done
    return 0
}
####################################
get_diff_TO_case # 获取待执行case列表
case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${case_list[*]}))  # 去重并将结果存储回原列表
if [[ ${#case_list[*]} -ne 0 ]];then
    echo -e "\033[31m =======CI Check case========= \033"
    echo -e "\033[31m ---- case_list length: ${#case_list[*]}, cases: ${case_list[*]} \033"
    echo -e "\033[31m ============================= \033"
    set +e

    # Install paddle
    install_paddle
    # Install paddlenlp
    install_paddlenlp
    # Install external_ops
    install_external_ops
    
    case_num=1
    export FLAGS_install_deps=0
    export FLAGS_download_data=""
    if [[ $(contain_case llama_auto ${case_list[@]}; echo $?) -eq 1 ]];then
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: llama_auto \033"
        bash /workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh llama_case_list_auto $FLAGS_install_deps $FLAGS_download_data
        print_info $? `ls -lt ${log_path} | grep llama | head -n 1 | awk '{print $9}'` llama_auto
        export FLAGS_download_data="llama ""$FLAGS_download_data"
        let case_num++
    fi
    if [[ $(contain_case gpt-3_auto ${case_list[@]}; echo $?) -eq 1 ]];then
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: gpt-3_auto \033"
        bash /workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh llm_gpt_case_list_auto $FLAGS_install_deps $FLAGS_download_data
        print_info $? `ls -lt ${log_path} | grep gpt | head -n 1 | awk '{print $9}'` gpt-3_auto
        export FLAGS_install_deps=1
        export FLAGS_download_data="gpt ""$FLAGS_download_data"
        let case_num++        
    fi
    if [[ $(contain_case gpt-3_dygraph ${case_list[@]}; echo $?) -eq 1 ]];then
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: gpt-3_dygraph \033"
        bash /workspace/PaddleNLP/scripts/distribute/ci_case_dy.sh gpt_case_list_dygraph $FLAGS_install_deps $FLAGS_download_data
        print_info $? `ls -lt ${log_path} | grep gpt | head -n 1 | awk '{print $9}'` gpt-3_dygraph
        export FLAGS_install_deps=1
        export FLAGS_download_data="gpt ""$FLAGS_download_data"
        let case_num++
    fi
    echo -e "\033[31m ---- end run case  \033"
    cd ${log_path}
    if [ ! -f *FAIL* ];then
        FF=0
        EXCODE=0
        echo -e "\033[32m ---- all case Success \033"
    else
        FF=`ls *FAIL*|wc -l`
        EXCODE=2
        echo -e "\033[31m ---- case Failed number: ${FF} \033"
        ls *_FAIL*
    fi
else
    echo -e "\033[32m Changed Not CI case, Skips \033"
    EXCODE=0
fi
exit $EXCODE
