// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"
#include "all_reduce.h"

namespace py = pybind11;

PYBIND11_MODULE(paddlenlp_ops, m) {
    /**
     * all_reduce.cu
     */
    m.def("init_custom_all_reduce", &init_custom_all_reduce, "init all reduce class function");
    m.def("all_reduce", &all_reduce, "all reduce function");
    m.def("dispose", &dispose, "del function for python");
    m.def("meta_size", &meta_size, "meta_size function for Signal struct");
    m.def("register_buffer", &register_buffer, "register ipc buffer");
}
