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
# build python lib
mkdir -p build_py36 build_py37 build_py38 build_py39 build_py310
for py_version in 6 7 8 9 10;
do
   cd build_py3${py_version}
   rm -rf *
   platform="$(uname -s)"
   if [[ $platform == Linux* ]];
   then
      export LD_LIBRARY_PATH=/opt/_internal/cpython-3.${py_version}.0/lib/:${LD_LIBRARY_PATH}
      export PATH=/opt/_internal/cpython-3.${py_version}.0/bin/:${PATH}
      core_num=`nproc`
   else
      export LD_LIBRARY_PATH=/Users/paddle/miniconda2/envs/py3${py_version}/lib/:${LD_LIBRARY_PATH}
      export PATH=/Users/paddle/miniconda2/envs/py3${py_version}/bin/:${PATH}
      core_num=`sysctl -n hw.logicalcpu`
   fi
   echo "Compile with $core_num cores"
   cmake .. -DWITH_PYTHON=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
   make -j${core_num}
   if [[ $? == 0 ]];
   then
       echo "Successfully compile."
   else
       echo "Fail compiling."
   fi
   cd ..
done

