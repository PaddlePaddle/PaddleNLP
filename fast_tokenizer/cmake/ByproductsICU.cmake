# MIT License
#
# Copyright (c) 2018 The ViaDuck Project
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

function(GetICUByproducts ICU_PATH ICU_LIB_VAR ICU_INCLUDE_VAR ICU_BASE_NAMES_VAR)
    # include directory
    set(${ICU_INCLUDE_VAR} "${ICU_PATH}/include" PARENT_SCOPE)
    
    if (WIN32)
        # windows basenames and pre/suffixes
        set(ICU_LIB_BASE_NAMES dt in io tu uc)
        
        set(ICU_SHARED_PREFIX "lib")
        set(ICU_STATIC_PREFIX "")
        set(ICU_SHARED_SUFFIX ".dll.a")
        set(ICU_STATIC_SUFFIX ".lib")
        set(ICU_INSTALL_LIB "lib64")
    else()
        # unix basenames and pre/suffixes
        set(ICU_LIB_BASE_NAMES i18n data uc io tu)
        set(ICU_SHARED_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
        set(ICU_STATIC_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
        set(ICU_SHARED_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(ICU_STATIC_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
        set(ICU_INSTALL_LIB "lib")
    endif()
    # add static and shared libs to the libraries variable
    foreach(ICU_BASE_NAME ${ICU_LIB_BASE_NAMES})
        set(ICU_SHARED_LIB "${ICU_PATH}/${ICU_INSTALL_LIB}/${ICU_SHARED_PREFIX}icu${ICU_BASE_NAME}${ICU_SHARED_SUFFIX}")
        set(ICU_STATIC_LIB "${ICU_PATH}/${ICU_INSTALL_LIB}/${ICU_STATIC_PREFIX}icu${ICU_BASE_NAME}${ICU_STATIC_SUFFIX}")
        
        if (ICU_STATIC)
            list(APPEND ${ICU_LIB_VAR} ${ICU_STATIC_LIB})
        else()
            list(APPEND ${ICU_LIB_VAR} ${ICU_SHARED_LIB})
        endif()
        list(APPEND ${ICU_BASE_NAMES_VAR} ${ICU_BASE_NAME})
    endforeach()
    set(${ICU_LIB_VAR} ${${ICU_LIB_VAR}} PARENT_SCOPE)
    set(${ICU_BASE_NAMES_VAR} ${${ICU_BASE_NAMES_VAR}} PARENT_SCOPE)
endfunction()