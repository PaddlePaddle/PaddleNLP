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

####################################
install_paddle(){
    echo -e "\033[31m ---- Install paddlepaddle-gpu  \033"
    python -m pip install --user ${paddle} --force-reinstall --no-dependencies;
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
}

install_paddlenlp(){
    echo -e "\033[31m ---- Install paddlenlp  \033"
    cd ${nlp_dir}
    sed -i -e "s/paddlenlp/#paddlenlp/g" model_zoo/gpt-3/requirements.txt
    export http_proxy=${proxy} && export https_proxy=${proxy}
    python -m pip uninstall paddlenlp -y
    rm -rf build/ && rm -rf paddlenlp.egg-info/ && rm -rf dist/
    python -m pip install --ignore-installed -r requirements.txt
    python setup.py install
    python setup.py build_ext
    python setup.py bdist_wheel
    unset http_proxy && unset https_proxy
    cd -
    python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)";
}
####################################
get_diff_TO_case(){
cd ${nlp_dir}
export FLAGS_paddlenlp=0
for file_name in `git diff --numstat upstream/${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    dir4=${arr_file_name[3]}
    echo "file_name:"${file_name}, "dir1:"${dir1}, "dir2:"${dir2},"dir3:"${dir3},"dir4:"${dir4},".xx:" ${file_name##*.}
    if [ ! -f ${file_name} ];then # 针对pr删掉文件
        continue
    elif [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
        continue
    elif [[ ${dir1} =~ "model_zoo" ]] && [[ ${dir2} =~ "gpt-3" ]];then
        if [[ ${dir3} =~ "benchmarks" ]];then
            continue
        else
            # model_zoo/gpt-3
            case_list[${#case_list[*]}]=gpt-3_auto
            case_list[${#case_list[*]}]=gpt-3_dygraph
        fi
    elif [[ ${dir1} =~ "paddlenlp" ]];then
        export FLAGS_paddlenlp=1
    else
        continue
    fi
done
}
####################################
print_info(){
if [ $1 -ne 0 ];then
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
get_diff_TO_case # 获取待执行case列表
case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${case_list[*]}))  # 去重并将结果存储回原列表
if [[ ${#case_list[*]} -ne 0 ]];then
    echo -e "\033[31m =======CI Check case========= \033"
    echo -e "\033[31m ---- case_list length: ${#case_list[*]}, cases: ${case_list[*]} \033"
    echo -e "\033[31m ============================= \033"
    set +e

    # Install paddle
    install_paddle
    if [[ FLAGS_paddlenlp -eq 1 ]];then
        # 安装本地paddlenlp
        install_paddlenlp
    fi
    case_num=1
    export FLAGS_before_hook=0
    for case in ${case_list[*]};do
        echo -e "\033[31m ---- running case $case_num/${#case_list[*]}: ${case} \033"
        if [[ ${case} == "gpt-3_auto" ]];then
            bash /workspace/PaddleNLP/scripts/distribute/ci_case_auto.sh case_list_auto $FLAGS_before_hook
            print_info $? `ls -lt ${log_path} | grep gpt | head -n 1 | awk '{print $9}'` ${case}
            export FLAGS_before_hook=1
            let case_num++
        elif [[ ${case} == "gpt-3_dygraph" ]];then
            bash /workspace/PaddleNLP/scripts/distribute/ci_case_dy.sh case_list_dygraph $FLAGS_before_hook
            print_info $? `ls -lt ${log_path} | grep gpt | head -n 1 | awk '{print $9}'` ${case}
            export FLAGS_before_hook=1
            let case_num++
        else
            echo -e "\033[31m ---- no ${case} \033"
            let case_num++
        fi
    done
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
