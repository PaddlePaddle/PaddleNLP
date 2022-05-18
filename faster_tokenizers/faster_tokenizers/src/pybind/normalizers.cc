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

#include <Python.h>

#include "normalizers/normalizers.h"
#include "pybind/normalizers.h"

namespace py = pybind11;

namespace tokenizers {
namespace pybind {

class PyNormalizer : public normalizers::Normalizer {
public:
  using Normalizer::Normalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        void, Normalizer, "__call__", operator(), mut_str);
  }
};

class PyBertNormalizer : public normalizers::BertNormalizer {
public:
  using BertNormalizer::BertNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, BertNormalizer, "__call__", operator(), mut_str);
  }
};

class PyReplaceNormalizer : public normalizers::ReplaceNormalizer {
public:
  using ReplaceNormalizer::ReplaceNormalizer;
  PyReplaceNormalizer(const ReplaceNormalizer& r) : ReplaceNormalizer(r) {}
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, ReplaceNormalizer, "__call__", operator(), mut_str);
  }
};

class PyStripNormalizer : public normalizers::StripNormalizer {
public:
  using StripNormalizer::StripNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, StripNormalizer, "__call__", operator(), mut_str);
  }
};

class PyStripAccentsNormalizer : public normalizers::StripAccentsNormalizer {
public:
  using StripAccentsNormalizer::StripAccentsNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, StripAccentsNormalizer, "__call__", operator(), mut_str);
  }
};

class PyNFCNormalizer : public normalizers::NFCNormalizer {
public:
  using NFCNormalizer::NFCNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, NFCNormalizer, "__call__", operator(), mut_str);
  }
};

class PyNFKCNormalizer : public normalizers::NFKCNormalizer {
public:
  using NFKCNormalizer::NFKCNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, NFKCNormalizer, "__call__", operator(), mut_str);
  }
};

class PyNFDNormalizer : public normalizers::NFDNormalizer {
public:
  using NFDNormalizer::NFDNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, NFDNormalizer, "__call__", operator(), mut_str);
  }
};

class PyNFKDNormalizer : public normalizers::NFKDNormalizer {
public:
  using NFKDNormalizer::NFKDNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, NFKDNormalizer, "__call__", operator(), mut_str);
  }
};

class PyNmtNormalizer : public normalizers::NmtNormalizer {
public:
  using NmtNormalizer::NmtNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, NmtNormalizer, "__call__", operator(), mut_str);
  }
};

class PySequenceNormalizer : public normalizers::SequenceNormalizer {
public:
  using SequenceNormalizer::SequenceNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, SequenceNormalizer, "__call__", operator(), mut_str);
  }
};

class PyLowercaseNormalizer : public normalizers::LowercaseNormalizer {
public:
  using LowercaseNormalizer::LowercaseNormalizer;
  virtual void operator()(
      normalizers::NormalizedString* mut_str) const override {
    PYBIND11_OVERLOAD_NAME(
        void, LowercaseNormalizer, "__call__", operator(), mut_str);
  }
};

void BindNormalizers(pybind11::module* m) {
  auto submodule = m->def_submodule("normalizers", "The normalizers module");
  py::class_<normalizers::NormalizedString>(submodule, "NormalizedString")
      .def(py::init<const std::string&>())
      .def(py::init<>())
      .def("__str__", &normalizers::NormalizedString::GetStr);
  py::class_<normalizers::Normalizer, PyNormalizer>(submodule, "Normalizer")
      .def(py::init<>())
      .def("normalize_str",
           [](const normalizers::Normalizer& self, const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::Normalizer::operator());
  py::class_<normalizers::BertNormalizer, PyBertNormalizer>(submodule,
                                                            "BertNormalizer")
      .def(py::init<bool, bool, bool, bool>(),
           py::arg("clean_text") = true,
           py::arg("handle_chinese_chars") = true,
           py::arg("strip_accents") = true,
           py::arg("lowercase") = true)
      .def("normalize_str",
           [](const normalizers::BertNormalizer& self, const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::BertNormalizer::operator())
      .def("__getstate__", [](const normalizers::BertNormalizer& self) {
        nlohmann::json j = self;
        return j.dump();
      });

  py::class_<normalizers::ReplaceNormalizer, PyReplaceNormalizer>(
      submodule, "ReplaceNormalizer")
      .def(py::init<const normalizers::ReplaceNormalizer&>(),
           py::arg("replace_normalizer"))
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("pattern"),
           py::arg("content"))
      .def("normalize_str",
           [](const normalizers::ReplaceNormalizer& self,
              const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::ReplaceNormalizer::operator());
  py::class_<normalizers::StripNormalizer, PyStripNormalizer>(submodule,
                                                              "StripNormalizer")
      .def(py::init<bool, bool>(),
           py::arg("left") = true,
           py::arg("right") = true)
      .def(
          "normalize_str",
          [](const normalizers::StripNormalizer& self, const std::string& str) {
            normalizers::NormalizedString normalized(str);
            self(&normalized);
            return normalized.GetStr();
          },
          py::arg("sequence"))
      .def("__call__", &normalizers::StripNormalizer::operator());
  py::class_<normalizers::StripAccentsNormalizer, PyStripAccentsNormalizer>(
      submodule, "StripAccentsNormalizer")
      .def(py::init<>())
      .def("normalize_str",
           [](const normalizers::StripAccentsNormalizer& self,
              const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::StripAccentsNormalizer::operator());
  py::class_<normalizers::NFCNormalizer, PyNFCNormalizer>(submodule,
                                                          "NFCNormalizer")
      .def(py::init<>())
      .def("normalize_str",
           [](const normalizers::NFCNormalizer& self, const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::NFCNormalizer::operator());
  py::class_<normalizers::NFDNormalizer, PyNFDNormalizer>(submodule,
                                                          "NFDNormalizer")
      .def(py::init<>())
      .def("normalize_str",
           [](const normalizers::NFDNormalizer& self, const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::NFDNormalizer::operator());
  py::class_<normalizers::NFKCNormalizer, PyNFKCNormalizer>(submodule,
                                                            "NFKCNormalizer")
      .def(py::init<>())
      .def("normalize_str",
           [](const normalizers::NFKCNormalizer& self, const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::NFKCNormalizer::operator());
  py::class_<normalizers::NFKDNormalizer, PyNFKDNormalizer>(submodule,
                                                            "NFKDNormalizer")
      .def(py::init<>())
      .def("normalize_str",
           [](const normalizers::NFKDNormalizer& self, const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::NFKDNormalizer::operator());
  py::class_<normalizers::NmtNormalizer, PyNmtNormalizer>(submodule,
                                                          "NmtNormalizer")
      .def(py::init<>())
      .def("normalize_str",
           [](const normalizers::NmtNormalizer& self, const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::NmtNormalizer::operator());
  py::class_<normalizers::LowercaseNormalizer, PyLowercaseNormalizer>(
      submodule, "LowercaseNormalizer")
      .def(py::init<>())
      .def("normalize_str",
           [](const normalizers::LowercaseNormalizer& self,
              const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::LowercaseNormalizer::operator());
  py::class_<normalizers::SequenceNormalizer, PySequenceNormalizer>(
      submodule, "SequenceNormalizer")
      .def(py::init<const std::vector<normalizers::Normalizer*>&>())
      .def("normalize_str",
           [](const normalizers::SequenceNormalizer& self,
              const std::string& str) {
             normalizers::NormalizedString normalized(str);
             self(&normalized);
             return normalized.GetStr();
           },
           py::arg("sequence"))
      .def("__call__", &normalizers::SequenceNormalizer::operator());
}

}  // pybind
}  // tokenizers
