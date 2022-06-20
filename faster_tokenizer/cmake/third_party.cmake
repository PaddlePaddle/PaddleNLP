# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

set(THIRD_PARTY_PATH  "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
    "A path setting third party libraries download & build directories.")

include(external/icu)
include(external/gtest)
include(external/gflags)
include(external/glog)
include(external/re2)
include(external/boost)
include(external/nlohmann_json)
include(external/dart) # For trie
if (WITH_PYTHON)
include(external/python)
include(external/pybind11)
endif()