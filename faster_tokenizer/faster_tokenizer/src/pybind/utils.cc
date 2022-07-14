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

#include <sstream>
#include <unordered_map>

#include "pybind/utils.h"
namespace py = pybind11;

namespace paddlenlp {
namespace faster_tokenizer {
namespace pybind {

PyObject* ToPyObject(bool value) {
  if (value) {
    Py_INCREF(Py_True);
    return Py_True;
  } else {
    Py_INCREF(Py_False);
    return Py_False;
  }
}

PyObject* ToPyObject(int value) { return PyLong_FromLong(value); }

PyObject* ToPyObject(uint32_t value) { return PyLong_FromUnsignedLong(value); }

PyObject* ToPyObject(int64_t value) { return PyLong_FromLongLong(value); }

PyObject* ToPyObject(size_t value) { return PyLong_FromSize_t(value); }

PyObject* ToPyObject(float value) { return PyLong_FromDouble(value); }

PyObject* ToPyObject(double value) { return PyLong_FromDouble(value); }

PyObject* ToPyObject(const char* value) { return PyUnicode_FromString(value); }

PyObject* ToPyObject(const std::string& value) {
  return PyUnicode_FromString(value.c_str());
}

PyObject* ToPyObject(const std::vector<bool>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<int>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<int64_t>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, (Py_ssize_t)i, ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<size_t>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, (Py_ssize_t)i, ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<float>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<double>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<std::vector<size_t>>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());

  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }

  return result;
}

PyObject* ToPyObject(const std::vector<std::string>& value) {
  PyObject* result = PyList_New((Py_ssize_t)value.size());
  for (size_t i = 0; i < value.size(); i++) {
    PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), ToPyObject(value[i]));
  }
  return result;
}

bool CastPyArg2AttrBoolean(PyObject* obj, ssize_t arg_pos) {
  if (obj == Py_None) {
    return false;  // To be compatible with QA integration testing. Some
                   // test case pass in None.
  } else if (obj == Py_True) {
    return true;
  } else if (obj == Py_False) {
    return false;
  } else {
    std::ostringstream oss;
    oss << "argument (position" << arg_pos + 1 << " must be bool, but got "
        << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
    throw std::runtime_error(oss.str());
  }
  return false;
}

std::string CastPyArg2AttrString(PyObject* obj, ssize_t arg_pos) {
  if (PyUnicode_Check(obj)) {
    Py_ssize_t size;
    const char* data;
    data = PyUnicode_AsUTF8AndSize(obj, &size);
    return std::string(data, static_cast<size_t>(size));
  } else {
    std::ostringstream oss;
    oss << "argument (position" << arg_pos + 1 << " must be str, but got "
        << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
    throw std::runtime_error(oss.str());
    return "";
  }
}

int CastPyArg2AttrInt(PyObject* obj, ssize_t arg_pos) {
  if (PyLong_Check(obj) && !PyBool_Check(obj)) {
    return static_cast<int>(PyLong_AsLong(obj));
  } else {
    std::ostringstream oss;
    oss << "argument (position" << arg_pos + 1 << " must be str, but got "
        << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
    throw std::runtime_error(oss.str());
    return 0;
  }
}

int64_t CastPyArg2AttrLong(PyObject* obj, ssize_t arg_pos) {
  if (PyLong_Check(obj) && !PyBool_Check(obj)) {
    return (int64_t)PyLong_AsLong(obj);  // NOLINT
  } else {
    std::ostringstream oss;
    oss << "argument (position" << arg_pos + 1 << " must be str, but got "
        << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
    throw std::runtime_error(oss.str());
    return 0;
  }
}

size_t CastPyArg2AttrSize_t(PyObject* obj, ssize_t arg_pos) {
  if (PyLong_Check(obj) && !PyBool_Check(obj)) {
    return PyLong_AsSize_t(obj);
  } else {
    std::ostringstream oss;
    oss << "argument (position" << arg_pos + 1 << " must be str, but got "
        << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
    throw std::runtime_error(oss.str());
    return 0;
  }
}

float CastPyArg2AttrFloat(PyObject* obj, ssize_t arg_pos) {
  if (PyFloat_Check(obj) || PyLong_Check(obj)) {
    return static_cast<float>(PyFloat_AsDouble(obj));
  } else {
    std::ostringstream oss;
    oss << "argument (position" << arg_pos + 1 << " must be str, but got "
        << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
    throw std::runtime_error(oss.str());
    return 0;
  }
}

std::vector<std::string> CastPyArg2VectorOfStr(PyObject* obj, size_t arg_pos) {
  std::vector<std::string> result;
  if (PyList_Check(obj)) {
    Py_ssize_t len = PyList_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyList_GetItem(obj, i);
      if (PyUnicode_Check(item)) {
        result.emplace_back(CastPyArg2AttrString(item, 0));
      } else {
        std::ostringstream oss;
        oss << "argument (position" << arg_pos + 1
            << " must be list of str, but got "
            << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
        throw std::runtime_error(oss.str());
        return {};
      }
    }
  } else if (PyTuple_Check(obj)) {
    Py_ssize_t len = PyTuple_Size(obj);
    PyObject* item = nullptr;
    for (Py_ssize_t i = 0; i < len; i++) {
      item = PyTuple_GetItem(obj, i);
      if (PyUnicode_Check(item)) {
        result.emplace_back(CastPyArg2AttrString(item, 0));
      } else {
        std::ostringstream oss;
        oss << "argument (position" << arg_pos + 1
            << " must be list of str, but got "
            << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
        throw std::runtime_error(oss.str());
        return {};
      }
    }
  } else if (obj == Py_None) {
    return {};
  } else {
    std::ostringstream oss;
    oss << "argument (position" << arg_pos + 1
        << " must be list or tuple, but got "
        << (reinterpret_cast<PyTypeObject*>(obj->ob_type))->tp_name;
    throw std::runtime_error(oss.str());
    return {};
  }
  return result;
}

bool PyObject_CheckLongOrConvertToLong(PyObject** obj) {
  if ((PyLong_Check(*obj) && !PyBool_Check(*obj))) {
    return true;
  }

  if (std::string((reinterpret_cast<PyTypeObject*>((*obj)->ob_type))->tp_name)
          .find("numpy") != std::string::npos) {
    auto to = PyNumber_Long(*obj);
    if (to) {
      *obj = to;
      return true;
    }
  }

  return false;
}

}  // namespace pybind
}  // namespace faster_tokenizer
}  // namespace paddlenlp
