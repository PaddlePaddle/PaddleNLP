/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace paddlenlp {
namespace faster_tokenizer {
namespace pybind {

PyObject* ToPyObject(int value);
PyObject* ToPyObject(uint32_t value);
PyObject* ToPyObject(bool value);
PyObject* ToPyObject(int64_t value);
PyObject* ToPyObject(size_t value);
PyObject* ToPyObject(float value);
PyObject* ToPyObject(double value);
PyObject* ToPyObject(const char* value);
PyObject* ToPyObject(const std::string& value);
PyObject* ToPyObject(const std::vector<bool>& value);
PyObject* ToPyObject(const std::vector<int>& value);
PyObject* ToPyObject(const std::vector<int64_t>& value);
PyObject* ToPyObject(const std::vector<size_t>& value);
PyObject* ToPyObject(const std::vector<float>& value);
PyObject* ToPyObject(const std::vector<double>& value);
PyObject* ToPyObject(const std::vector<std::vector<size_t>>& value);
PyObject* ToPyObject(const std::vector<std::string>& value);

bool PyObject_CheckLongOrConvertToLong(PyObject** obj);
bool CastPyArg2AttrBoolean(PyObject* obj, ssize_t arg_pos);
std::string CastPyArg2AttrString(PyObject* obj, ssize_t arg_pos);
int CastPyArg2AttrInt(PyObject* obj, ssize_t arg_pos);
int64_t CastPyArg2AttrLong(PyObject* obj, ssize_t arg_pos);
size_t CastPyArg2AttrSize_t(PyObject* obj, ssize_t arg_pos);
float CastPyArg2AttrFloat(PyObject* obj, ssize_t arg_pos);
std::vector<std::string> CastPyArg2VectorOfStr(PyObject* obj, size_t arg_pos);

template <typename T>
std::vector<T> CastPyArg2VectorOfInt(PyObject* obj, size_t arg_pos) {
  std::vector<T> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyObject_CheckLongOrConvertToLong(&item)) {
        result.emplace_back(static_cast<T>(PyLong_AsLong(item)));
      } else {
        std::ostringstream oss;
        oss << "argument (position " << arg_pos + 1
            << "must be list of int, but got "
            << reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name
            << " at pos " << i;
        throw oss.str();
        return {};
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyObject_CheckLongOrConvertToLong(&item)) {
        result.emplace_back(static_cast<T>(PyLong_AsLong(item)));
      } else {
        std::ostringstream oss;
        oss << "argument (position " << arg_pos + 1
            << "must be list of int, but got "
            << reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name
            << " at pos " << i;
        throw oss.str();
        return {};
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else {
    std::ostringstream oss;
    oss << "argument (position " << arg_pos + 1
        << "must be list or tuple, but got "
        << reinterpret_cast<PyTypeObject*>(obj->ob_type)->tp_name;
    throw oss.str();
    return {};
  }
  return result;
}
}  // namespace pybind
}  // namespace faster_tokenizer
}  // namespace paddlenlp
