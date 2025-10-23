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

failed_num=0
echo_list=()
approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/PaddleNLP/pulls/${PR_ID}/reviews?per_page=10000`

function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}

function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    echo ${person_num}
    APPROVALS=`echo ${approval_line}|python check_pr_approval.py $1 $person_num`
    echo ${APPROVALS}
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}

echo_line="The PaddleNLP repository will be switched to the PaddleFormers repository soon, so the PR needs to be merged into PaddleFormers first and the PR link should be filled in the current PR description area. Then please contact From00 for approval."
check_approval 1 From00

if [ -n "${echo_list}" ];then
    echo "**************************************************************"
    echo "Please find RD for approval."
    echo -e "${echo_list[@]}"
    echo "There are ${failed_num} approved errors."
    echo "**************************************************************"
    exit 1
else
    echo "**************************************************************"
    echo "CI APPROVAL PASSED."
    echo "**************************************************************"
    exit 0
fi
