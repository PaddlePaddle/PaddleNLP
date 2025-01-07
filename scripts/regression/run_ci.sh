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
export python=$1
export paddle=$2
export nlp_dir=/workspace/PaddleNLP
mkdir /workspace/PaddleNLP/logs
mkdir /workspace/PaddleNLP/model_logs
mkdir /workspace/PaddleNLP/unittest_logs
mkdir /workspace/PaddleNLP/coverage_logs
mkdir /workspace/PaddleNLP/upload
export log_path=/workspace/PaddleNLP/model_logs
export P0case_list=()
export APIcase_list=()
declare -A Normal_dic
declare -A all_P0case_dic
declare -A Build_list
target_lists_for_llm=(
    "paddlenlp/transformers"
    "paddlenlp/experimental/transformers/"
    "paddlenlp/data"
    "paddlenlp/datasets"
    "paddlenlp/generation"
    "paddlenlp/peft"
    "paddlenlp/mergekit"
    "paddlenlp/quantization"
    "paddlenlp/trainer"
    "paddlenlp/trl"
    "llm"
    "tests/llm"
    "csrc"
    "scripts/regression"
)
all_P0case_dic=(["msra_ner"]=15 
    ["glue"]=2 
    ["bert"]=2 
    ["skep"]=10 
    ["bigbird"]=2
    ["gpt"]=2 
    ["ernie-1.0"]=2 
    ["xlnet"]=2
    ["ofa"]=2 ["albert"]=2 ["lexical_analysis"]=5
    ["transformer"]=5
    ["question_matching"]=5 ["ernie-csc"]=5  ["taskflow"]=5 ["clue"]=5 ["textcnn"]=5
    ["ernie-3.0"]=5 ["ernie-layout"]=5 ["uie"]=5  ["llm"]=5
    ["ernie"]=2 ["ernie_layout"]=5 ["ernie_csc"]=5 ["ernie_ctm"]=5 ["segment_parallel_utils"]=5 ["ring_flash_attention"]=5)
####################################

python -m pip config --user set global.index http://pip.baidu-int.com/search/
python -m pip config --user set global.index-url http://pip.baidu-int.com/simple
python -m pip config --user set global.trusted-host pip.baidu-int.com
# Insatll paddlepaddle-gpu
install_paddle(){
    echo -e "\033[35m ---- Install paddlepaddle-gpu  \033[0m"
    python -m pip install --user -r scripts/regression/requirements_ci.txt
    python -m pip uninstall paddlepaddle -y
    python -m pip install pillow -y
    python -m pip install --user ${paddle} --no-cache-dir;
    python -c "import paddle;print('paddle');print(paddle.__version__);print(paddle.version.show())" >> ${log_path}/commit_info.txt
    python -c 'from visualdl import LogWriter'
}
####################################
# Install paddlenlp func
nlp_build (){
    cd $1
    rm -rf build/
    rm -rf paddlenlp.egg-info/
    rm -rf ppdiffusers.egg-info/
    rm -rf paddle_pipelines.egg-info/
    rm -rf dist/

    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python setup.py bdist_wheel
    # python -m pip install --ignore-installed  dist/p****.whl
}
####################################
# upload paddlenlp  whl
upload (){
    mkdir ${PPNLP_HOME}/upload
    if [ $1 == "paddlenlp" ];then
        echo -e "\033[35m ---- build latest paddlenlp  \033[0m"
        build_dev_path=/workspace/PaddleNLP_dev
        nlp_build ${build_dev_path}
        nlp_version=$(python -c "from paddlenlp import __version__; print(__version__)")
        # for test https://www.paddlepaddle.org.cn/whl/paddlenlp.html
        cp $build_dev_path/dist/p****.whl ${PPNLP_HOME}/upload/
        # for ci pr test
        cp $build_dev_path/dist/p****.whl ${PPNLP_HOME}/upload/paddlenlp-ci-py3-none-any.whl
        echo -e "\033[35m ---- build ${GIT_PR_ID} paddlenlp  \033[0m"
        build_pr_path=${nlp_dir}
        nlp_build ${build_pr_path}
    fi
}
####################################
# get diff case
for line in `cat scripts/regression/model_list.txt`;do
    all_example_dict[${#all_example_dict[*]}]=$line
done
cd ${nlp_dir}
get_diff_TO_P0case(){
for file_name in `git diff --numstat ${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    dir4=${arr_file_name[3]}
    file_item=$dir1/$dir2/$dir3/$dir4
    echo "file_name:"${file_name}, "dir1:"${dir1}, "dir2:"${dir2},"dir3:"${dir3},".xx:" ${file_name##*.}
    if [ ! -f ${file_name} ];then # 针对pr删掉文件
        continue
    elif [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
        continue
    elif [[ "${AGILE_COMPILE_BRANCH}" == "refactor-training-loop" ]];then
        P0case_list[${#P0case_list[*]}]=gpt
    elif [[ ${dir1} =~ "scripts" ]];then # API 升级
        if [[ ${dir2} =~ "should_deploy" ]];then # 针对发版mini test
            P0case_list[${#P0case_list[*]}]=transformer
        fi
        if [[ ${dir2} =~ "regression" ]];then # ci脚本修改
            P0case_list[${#P0case_list[*]}]=llm
        fi
    elif [[ ${dir1} =~ "paddlenlp" ]];then # API 升级
        for ((i=0; i<${#target_lists_for_llm[@]}; i++)); do  # 命中指定路径执行llm
            if [[ ${file_item} == *${target_lists_for_llm[i]}* ]];then
                P0case_list[${#P0case_list[*]}]=llm
            fi
        done
        if [[ ${dir2} =~ "__init__" ]];then # 针对发版mini test
            P0case_list[${#P0case_list[*]}]=bert
        elif [[ ${!all_P0case_dic[*]} =~ ${dir2} ]];then
            P0case_list[${#P0case_list[*]}]=${dir2}
        elif [[ ${dir2} =~ "transformers" ]];then
            P0case_list[${#P0case_list[*]}]=llm
            if [[ ${!all_P0case_dic[*]} =~ ${dir3} ]];then
                P0case_list[${#P0case_list[*]}]=${dir3}
            fi
        elif [[ ${dir2} =~ "taskflow" ]];then
            P0case_list[${#P0case_list[*]}]=taskflow
        elif [[ ${dir3} =~ "transformers" ]];then
            P0case_list[${#P0case_list[*]}]=llm
        fi
        Build_list[${dir1}]="paddlenlp" # 影响编包
    elif [[ ${dir1} =~ "examples" ]];then # 模型升级
        if [[ ${!all_P0case_dic[*]} =~ ${dir2} ]];then
            P0case_list[${#P0case_list[*]}]=${dir2}
        elif [[ ${!all_P0case_dic[*]} =~ ${dir3} ]];then
            P0case_list[${#P0case_list[*]}]=${dir3}
        elif [[ ${dir3##*.} == "py" ]] && [[ !(${all_example_dict[*]} =~ ${dir2}) ]];then #新增规范模型
            P0case_list[${#P0case_list[*]}]=${dir2}
            Normal_dic[${dir2}]="${dir1}/${dir2}/"
        elif [[ !(${all_example_dict[*]} =~ ${dir3}) ]] ;then
            P0case_list[${#P0case_list[*]}]=${dir3}
            Normal_dic[${dir3}]="${dir1}/${dir2}/${dir3}"
        fi
    elif [[ ${dir1} =~ "model_zoo" ]];then # 模型升级
        if [[ ${!all_P0case_dic[*]} =~ ${dir2} ]];then
            P0case_list[${#P0case_list[*]}]=${dir2}
        # elif [[ !(${all_example_dict[*]} =~ ${dir2}) ]];then #新增规范模型
        #     P0case_list[${#P0case_list[*]}]=${dir2}
        #     Normal_dic[${dir2}]="${dir1}/${dir2}/"
        fi
    elif [[ ${dir1} =~ "llm" ]];then # 模型升级
        P0case_list[${#P0case_list[*]}]=llm
    elif [[ ${dir1} =~ "tests" ]];then # 新增单测
        if [[ ${dir2} =~ "transformers" ]] ;then
            if [[ ${dir3##*.} == "py" ]];then
                continue
            elif [[ ${!all_P0case_dic[*]} =~ ${dir3} ]];then
                P0case_list[${#P0case_list[*]}]=${dir3}
            else
                APIcase_list[${#APIcase_list[*]}]=${dir3}
            fi
        elif [[ ${dir2} =~ "taskflow" ]] ;then
            APIcase_list[${#APIcase_list[*]}]=${dir2}
        elif [[ ${dir2} =~ "llm" ]] ;then
            P0case_list[${#P0case_list[*]}]=${dir2}
        fi
    elif [[ ${dir1} =~ "pipelines" ]];then # 影响编包
        Build_list[${dir1}]=${dir1}
    elif [[ ${dir1} =~ "ppdiffusers" ]];then # 影响编包
        Build_list[${dir1}]=${dir1}
    elif [[ ${dir1} =~ "csrc" ]];then # 推理改动
        P0case_list[${#P0case_list[*]}]=llm
    else
        continue
    fi
done
}
get_diff_TO_P0case
P0case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${P0case_list[*]}))
APIcase_list=($(awk -v RS=' ' '!a[$1]++' <<< ${APIcase_list[*]}))
####################################
# upload latest paddlenlp pipelines ppddifusers whl
if [[ ${#Build_list[*]} -ne 0 ]];then
    echo -e "\033[32m start build ${Build_list[*]} whl \033[0m"
    install_paddle
    for build_pkg in ${Build_list[*]};do
        upload ${build_pkg}
    done
    echo -e "\033[32m make PaddleNLP.tar.gz  \033[0m"
    cd /workspace
    rm -rf PaddleNLP_dev/build/*
    cd PaddleNLP_dev && git submodule update --init --recursive
    cd /workspace && tar -zcvf PaddleNLP.tar.gz PaddleNLP_dev/
    mv PaddleNLP.tar.gz ${PPNLP_HOME}/upload
    cd ${PPNLP_HOME}
    python upload.py ${PPNLP_HOME}/upload 'paddlenlp/wheels'
    rm -rf upload/*
else
   echo -e "\033[32m Don't need build whl  \033[0m"
fi
###################################
if [[ ${#P0case_list[*]} -ne 0 ]] || [[ ${#APIcase_list[*]} -ne 0 ]];then
    # Install paddlenlp
    cd ${nlp_dir}
    python -m pip uninstall protobuf -y
    python -m pip uninstall protobuf -y
    python -m pip install protobuf==3.20.2
    if [ ! -f ./dist/p****.whl ];then
        install_paddle
        echo "install_nlp_develop"
        wget https://paddlenlp.bj.bcebos.com/wheels/paddlenlp-ci-py3-none-any.whl
        python -m pip install --user paddlenlp-ci-py3-none-any.whl
    else
        echo "instal_nlp_pr"
        python -m pip install  dist/p****.whl
    fi
    python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)" >> ${log_path}/commit_info.txt
    python -m pip list
    echo -e "\033[35m =======CI Check P0case========= \033[0m"
    echo -e "\033[35m ---- P0case_list length: ${#P0case_list[*]}, cases: ${P0case_list[*]} \033[0m"
    set +e
    echo -e "\033[35m ---- start run P0case  \033[0m"
    case_num=1
    for p0case in ${P0case_list[*]};do
        echo -e "\033[35m ---- running P0case $case_num/${#P0case_list[*]}: ${p0case} \033[0m"
        if [[ ${!Normal_dic[*]} =~ ${p0case} ]];then
            # python ${nlp_dir}/scripts/regression/ci_normal_case.py ${Normal_dic[${p0case}]}
            # let case_num++
            echo "pass"
        else
            bash ${nlp_dir}/scripts/regression/ci_case.sh ${p0case} ${cudaid1} ${cudaid2}
            let case_num++
        fi
    done
    echo -e "\033[35m ---- end run P0case  \033[0m"
    cd ${nlp_dir}/model_logs
    FF=`ls *FAIL*|wc -l`
    EXCODE=0
    if [ "${FF}" -gt "0" ];then
        P0case_EXCODE=1
        EXCODE=2
    else
        P0case_EXCODE=0
    fi
    if [ $P0case_EXCODE -ne 0 ] ; then
        echo -e "\033[31m ---- P0case Failed number: ${FF} \033[0m"
        ls *_FAIL*
    else
        echo -e "\033[32m ---- P0case Success \033[0m"
    fi
    ####################################
    # run unittest
    cd ${nlp_dir}
    echo -e "\033[35m =======CI Check Unittest========= \033[0m"
    echo -e "\033[35m ---- unittest length: ${#APIcase_list[*]}, unittest cases: ${APIcase_list[*]} \033[0m"
    for apicase in ${APIcase_list[*]};do
        if [[ ${apicase} =~ "taskflow" ]] ; then
            python -m pytest tests/taskflow/test_*.py >${nlp_dir}/unittest_logs/${apicase}_unittest.log 2>&1
        else
            python -m pytest tests/transformers/${apicase}/test_*.py  >${nlp_dir}/unittest_logs/${apicase}_unittest.log 2>&1
            # sh run_coverage.sh paddlenlp.transformers.${apicase} >unittest_logs/${apicase}_coverage.log 2>&1
        fi
        UT_EXCODE=$? || true
        if [ $UT_EXCODE -ne 0 ] ; then
            mv ${nlp_dir}/unittest_logs/${apicase}_unittest.log ${nlp_dir}/unittest_logs/${apicase}_unittest_FAIL.log
        fi
    done
    cd ${nlp_dir}/unittest_logs
    UF=`ls *FAIL*|wc -l`
    if [ "${UF}" -gt "0" ];then
        UT_EXCODE=1
        EXCODE=3
    else
        UT_EXCODE=0
    fi
    if [ $UT_EXCODE -ne 0 ] ; then
        echo -e "\033[31m ---- Unittest Failed \033[0m"
        ls *_FAIL*
    else
        echo -e "\033[32m ---- Unittest Success \033[0m"
    fi
    cd ${nlp_dir}
    echo -e "\033[35m ---- Genrate Allure Report  \033[0m"
    cp scripts/regression/gen_allure_report.py ./
    python gen_allure_report.py
    echo -e "\033[35m ---- Report: https://xly.bce.baidu.com/ipipe/ipipe-report/report/${AGILE_JOB_BUILD_ID}/report/  \033[0m"
    ####################################
    # run coverage
    # cd ${nlp_dir}/tests/
    # bash run_coverage.sh
    # Coverage_EXCODE=$? || true
    # mv ./htmlcov ${nlp_dir}/coverage_logs/
    # if [ $Coverage_EXCODE -ne 0 ] ; then
    #     echo -e "\033[31m ---- Coverage Failed \033[0m"
    # else
    #     echo -e "\033[32m ---- Coverage Success \033[0m"
    # fi
    ####################################
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    EXCODE=0
fi
exit $EXCODE
