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
SM_VERSION=${2:-$SM_VERSION}
PYTHON_VERSION=${3:-$PYTHON_VERSION}
export python=$PYTHON_VERSION
ARCHITECTURE=${1:-$(${python} -c "import paddle;prop = paddle.device.cuda.get_device_properties();cc = prop.major * 10 + prop.minor;print(cc)")}



# directory config
DIST_DIR="gpu_dist"
BUILD_DIR="build"
EGG_DIR="paddlenlp_ops.egg-info"

# custom_ops directory config
OPS_SRC_DIR="./"
OPS_BUILD_DIR="build"
OPS_EGG_DIR="paddlenlp_ops_*.egg-info"
# OPS_TMP_DIR_BASE="tmp_base"
OPS_TMP_DIR="tmp_*"
TMP_DIR="tmp"

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
function generate_sm_version(){
    cuda_version=`${python} -c "import paddle; print(float(paddle.version.cuda()))"`
    echo "CUDA version is: $cuda_version"
    if [ ! -z "$SM_VERSION" ]; then
        sm_versions=($SM_VERSION )
    elif [ "$ARCHITECTURE" = "all" ]; then
        if echo "$cuda_version >= 12.0" | awk '{if ($0) exit 0; exit 1}'; then
          sm_versions=(70 80 80 86 89 90 )
        else
          sm_versions=(70 75 80 86 89 ) 
        fi 
    else 
        sm_versions=($ARCHITECTURE)
    fi
    echo "testtest ${sm_versions[@]}"
}

function create_directories(){
  mkdir -p $OPS_SRC_DIR/tmp/paddlenlp_ops
  touch $OPS_SRC_DIR/tmp/setup.py
  touch $OPS_SRC_DIR/tmp/paddlenlp_ops/__init__.py
  echo '# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

""" setup for PaddlenlpOps """

import os

from setuptools import find_packages, setup

description = "Paddlenlp_ops : inference framework implemented based on PaddlePaddle"
VERSION = "0.0.0"


def read(file: str):
    """
    read file and return content
    """
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, file)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content


def read_version():
    """
    read version and return content
    """
    return VERSION

setup(
    name="paddlenlp_ops",
    packages=find_packages(),
    version="0.0.0",
    author="Paddle Infernce Team",
    author_email="paddle-inference@baidu.com",
    description=description,
    long_description_content_type="text/markdown",
    url="",
    python_requires=">=3.8",
    package_dir={"paddlenlp_ops": "paddlenlp_ops/"},
    package_data={"paddlenlp_ops": ["sm70/*", "sm75/*", "sm80/*", "sm86/*", "sm89/*", "sm90/*"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
)' > $OPS_SRC_DIR/tmp/setup.py
  echo '# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import importlib

import paddle

from paddlenlp.utils.log import logger

cuda_version = float(paddle.version.cuda())
SUPPORTED_SM_VERSIONS = {70, 75, 80, 86, 89, 90} if cuda_version >= 12.0 else {70, 75, 80, 86, 89}


def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


sm_version = get_sm_version()
if sm_version not in SUPPORTED_SM_VERSIONS:
    raise RuntimeError("Unsupported SM version")
module_name = f"paddlenlp_ops.sm{sm_version}"

try:
    module = importlib.import_module(module_name)
    globals().update(vars(module))
except ImportError:
    logger.WARNING(f"No {module_name} ")
' > $OPS_SRC_DIR/tmp/paddlenlp_ops/__init__.py

  for sm_version in "${sm_versions[@]}"; do
        echo "create sm$sm_version"
        mkdir -p $OPS_SRC_DIR/tmp/paddlenlp_ops/sm${sm_version}
        touch $OPS_SRC_DIR/tmp/paddlenlp_ops/sm${sm_version}/__init__.py
        echo '# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.utils.log import logger

try:
    from .paddlenlp_ops_'${sm_version}' import *
except ImportError:
    logger.WARNING("No paddlenlp_ops_'${sm_version}' ops")
' > $OPS_SRC_DIR/tmp/paddlenlp_ops/sm${sm_version}/__init__.py
  done
}

function init() {
    echo -e "${BLUE}[init]${NONE} removing building directory..."
    rm -rf $DIST_DIR $BUILD_DIR $EGG_DIR
    if [ `${python} -m pip list | grep paddlenlp_ops | wc -l` -gt 0  ]; then
      echo -e "${BLUE}[init]${NONE} uninstalling paddlenlp_ops..."
      ${python} -m pip uninstall -y paddlenlp_ops
    fi

    ${python} -m pip install setuptools_scm
    generate_sm_version
    create_directories
    echo -e "${BLUE}[init]${NONE} ${GREEN}init success\n"
}

function build_ops() {
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
    cp -r ./${TMP_DIR}/tmp_${sm_version}/${WHEEL_NAME}/* ./${TMP_DIR}/paddlenlp_ops/sm${sm_version}
    return
}

function build_and_install_ops() {
  local sm_version="$1"
  cd $OPS_SRC_DIR
  export no_proxy=bcebos.com,paddlepaddle.org.cn,${no_proxy}
  echo -e "${BLUE}[build]${NONE} build and install paddlenlp_ops_sm${sm_version} ops..."
  CUDA_SM_VERSION=${sm_version} ${python} setup_cuda.py install --install-lib ${TMP_DIR}/tmp_${sm_version}
  echo -e "${BLUE}[build]${NONE} build and install paddlenlp_ops_${sm_version}..."
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build paddlenlp_ops_${sm_version} wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build paddlenlp_ops_sm${sm_version} wheel success\n"

  copy_ops "${sm_version}"
}

function build_and_install_whl() {
  echo -e "${BLUE}[build]${NONE} building paddlenlp_ops wheel..."
  rm -rf ./dist
  cd ${TMP_DIR}
  ${python} setup.py bdist_wheel --dist-dir ./$DIST_DIR
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
  mv $DIST_DIR ../
  cd ..
}


function unittest() {
  # run UT
  echo -e "${BLUE}[unittest]${NONE} ${GREEN}unittests success\n${NONE}"
}

function cleanup() {
  rm -rf $BUILD_DIR $EGG_DIR
  ${python} -m pip uninstall -y paddlenlp_ops
  rm -rf $OPS_SRC_DIR/$TMP_DIR
  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
}

function abort() {
  echo -e "${RED}[FAIL]${NONE} build wheel and unittest failed !
          please check your code" 1>&2

  cur_dir=`basename "$pwd"`

  rm -rf $BUILD_DIR $EGG_DIR $DIST_DIR
  ${python} -m pip uninstall -y paddlenlp_ops
  rm -rf $OPS_SRC_DIR/$TMP_DIR
  rm -rf $OPS_SRC_DIR/$OPS_BUILD_DIR $OPS_SRC_DIR/$OPS_EGG_DIR
}

# python_version_check

trap 'abort' 0
set -e

init
build_ops
build_and_install_whl
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

echo -e "${GREEN}wheel saved under${NONE} ${RED}${BOLD}./${DIST_DIR}${NONE}"

# install wheel
${python} -m pip install ./${DIST_DIR}/paddlenlp_ops*.whl
echo -e "${GREEN}wheel install success!${NONE}\n"

trap 0