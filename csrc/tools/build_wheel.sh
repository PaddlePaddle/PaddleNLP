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

PYTHON_VERSION=python
PYTHON_VERSION=${1:-$PYTHON_VERSION}
SM_VERSION=${2:-$SM_VERSION}
export python=$PYTHON_VERSION

# directory config
DIST_DIR="dist"
BUILD_DIR="build"
EGG_DIR="paddlenlp_ops.egg-info"

# custom_ops directory config
OPS_SRC_DIR="./"
OPS_BUILD_DIR="build"
OPS_EGG_DIR="paddlenlp_ops_*.egg-info"
# OPS_TMP_DIR_BASE="tmp_base"
OPS_TMP_DIR="tmp_*"

# TEST_DIR="tests"

# command line log config
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[1;32m'
BOLD='\033[1m'
NONE='\033[0m'


function python_version_check() {
  PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
  PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
  echo -e "find python version ${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
  if [ $PY_MAIN_VERSION -ne "3" -o $PY_SUB_VERSION -lt "8" ]; then
    echo -e "${RED}FAIL:${NONE} please use Python >= 3.8 !"
    exit 1
  fi
}

function init() {
    echo -e "${BLUE}[init]${NONE} removing building directory..."
    rm -rf $DIST_DIR $BUILD_DIR $EGG_DIR
    if [ `${python} -m pip list | grep paddlenlp_ops | wc -l` -gt 0  ]; then
      echo -e "${BLUE}[init]${NONE} uninstalling paddlenlp_ops..."
      ${python} -m pip uninstall -y paddlenlp_ops
    fi

    ${python} -m pip install setuptools_scm
    echo -e "${BLUE}[init]${NONE} ${GREEN}init success\n"
}

function generate_sm_versions_and_build_ops() {
   cuda_version=`${python} -c "import paddle; print(float(paddle.version.cuda()))"`
   echo "CUDA version is: $cuda_version"
   if [ ! -z "$SM_VERSION" ]; then
      sm_versions=($SM_VERSION )
   elif echo "$cuda_version >= 12.4" | awk '{if ($0) exit 0; exit 1}'; then
       sm_versions=(70 80 80 86 89 90 )
   else
       sm_versions=(70 75 80 86 89 ) 
    fi 
    
    for sm_version in "${sm_versions[@]}"; do
        echo "Building and installing for sm_version: $sm_version"
        build_and_install_ops $sm_version
    done
    return 
}

function copy_ops(){
    local sm_version="$1"
    OPS_VERSION="0.0.0"
    PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
    PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
    PY_VERSION="py${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
    SYSTEM_VERSION=`${python} -c "import platform; print(platform.system().lower())"`
    PROCESSER_VERSION=`${python} -c "import platform; print(platform.processor())"`
    WHEEL_NAME="paddlenlp_ops_${sm_version}-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSER_VERSION}.egg"
    echo -e "gpu ops -- paddlenlp_ops_${sm_version} ..."
    cp -r ./tmp_${sm_version}/${WHEEL_NAME}/* ./paddlenlp_ops/sm${sm_version}
    return
}

function build_and_install_ops() {
  local sm_version="$1"
  cd $OPS_SRC_DIR
  export no_proxy=bcebos.com,paddlepaddle.org.cn,${no_proxy}
  echo -e "${BLUE}[build]${NONE} build and install paddlenlp_ops_sm${sm_version} ops..."
  CUDA_SM_VERSION=${sm_version} ${python} setup_cuda.py install --install-lib tmp_${sm_version}
  echo -e "${BLUE}[build]${NONE} build and install paddlenlp_ops_${sm_version}..."
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build paddlenlp_ops_${sm_version} wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build paddlenlp_ops_sm${sm_version} wheel success\n"

  copy_ops "${sm_version}"
}

function build_and_install() {
  echo -e "${BLUE}[build]${NONE} building paddlenlp_ops wheel..."
  ${python} setup.py bdist_wheel
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build paddlenlp_ops wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build paddlenlp_ops wheel success\n"

  echo -e "${BLUE}[install]${NONE} installing paddlenlp_ops..."
  cd $DIST_DIR
  find . -name "paddlenlp_ops*.whl" | xargs ${python} -m pip install
  if [ $? -ne 0 ]; then
    cd ..
    echo -e "${RED}[FAIL]${NONE} install paddlenlp_ops wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[install]${NONE} ${GREEN}paddlenlp_ops install success\n"
  cd ..
}


function unittest() {
  # run UT
  echo -e "${BLUE}[unittest]${NONE} ${GREEN}unittests success\n${NONE}"
}

function cleanup() {
  rm -rf $BUILD_DIR $EGG_DIR
  ${python} -m pip uninstall -y paddlenlp_ops

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR $OPS_SRC_DIR/$OPS_TMP_DIR
}

function abort() {
  echo -e "${RED}[FAIL]${NONE} build wheel and unittest failed !
          please check your code" 1>&2

  cur_dir=`basename "$pwd"`

  rm -rf $BUILD_DIR $EGG_DIR $DIST_DIR
  ${python} -m pip uninstall -y paddlenlp_ops

  rm -rf $OPS_SRC_DIR/$OPS_BUILD_DIR $OPS_SRC_DIR/$OPS_EGG_DIR $OPS_SRC_DIR/$OPS_TMP_DIR
}

python_version_check

trap 'abort' 0
set -e

init
generate_sm_versions_and_build_ops
build_and_install
unittest
cleanup

# get Paddle version
PADDLE_VERSION=`${python} -c "import paddle; print(paddle.version.full_version)"`
PADDLE_COMMIT=`${python} -c "import paddle; print(paddle.version.commit)"`

# get paddlenlp_ops version
EFFLLM_BRANCH=`git rev-parse --abbrev-ref HEAD`
EFFLLM_COMMIT=`git rev-parse --short HEAD`

# get Python version
PYTHON_VERSION=`${python} -c "import platform; print(platform.python_version())"`

echo -e "\n${GREEN}paddlenlp_ops wheel compiled and checked success !${NONE}
        ${BLUE}Python version:${NONE} $PYTHON_VERSION
        ${BLUE}Paddle version:${NONE} $PADDLE_VERSION ($PADDLE_COMMIT)
        ${BLUE}paddlenlp_ops branch:${NONE} $EFFLLM_BRANCH ($EFFLLM_COMMIT)\n"

echo -e "${GREEN}wheel saved under${NONE} ${RED}${BOLD}./dist${NONE}"

# install wheel
${python} -m pip install ./dist/paddlenlp_ops*.whl
echo -e "${GREEN}wheel install success!${NONE}\n"

trap 0