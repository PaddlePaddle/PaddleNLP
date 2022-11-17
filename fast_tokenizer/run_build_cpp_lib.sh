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

# Can be used in linux and mac
mkdir -p build_cpp
cd build_cpp
rm -rf *
platform="$(uname -s)"
if [[ $platform == Linux* ]];
then
  core_num=`nproc`
else
  core_num=`sysctl -n hw.logicalcpu`
fi
echo "Compile with $core_num cores"
cmake .. -DWITH_PYTHON=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
make -j${core_num}

if [[ $? == 0 ]];
then
    echo "Successfully compile."
else
    echo "Fail compiling."
fi
cd ..
