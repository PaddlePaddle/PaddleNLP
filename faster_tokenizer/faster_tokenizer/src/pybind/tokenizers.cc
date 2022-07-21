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

#include <unordered_map>

#include "core/tokenizer.h"
#include "decoders/decoders.h"
#include "glog/logging.h"
#include "models/models.h"
#include "normalizers/normalizers.h"
#include "postprocessors/postprocessors.h"
#include "pretokenizers/pretokenizers.h"

#include <Python.h>

#include "pybind/exception.h"
#include "pybind/tokenizers.h"
#include "pybind/utils.h"

namespace py = pybind11;

namespace paddlenlp {
namespace faster_tokenizer {
namespace pybind {

PyTypeObject* p_tokenizer_type;  // For Tokenizer

PyNumberMethods number_methods;
PySequenceMethods sequence_methods;
PyMappingMethods mapping_methods;

typedef struct {
  PyObject_HEAD core::Tokenizer tokenizer;
  // Weak references
  PyObject* weakrefs;
} TokenizerObject;

static PyObject* TokenizerPropertiesGetNormaizer(TokenizerObject* self,
                                                 void* closure) {
  py::object py_obj = py::cast(self->tokenizer.GetNormalizerPtr());
  py_obj.inc_ref();
  return py_obj.ptr();
}

static int TokenizerPropertiesSetNormalizer(TokenizerObject* self,
                                            PyObject* value,
                                            void* closure) {
  TOKENIZERS_TRY
  py::handle py_obj(value);
  int ret = 0;
  if (pybind11::type::of(py_obj).is(
          py::type::of<normalizers::LowercaseNormalizer>())) {
    const auto& normalizer =
        py_obj.cast<const normalizers::LowercaseNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::BertNormalizer>())) {
    const auto& normalizer = py_obj.cast<const normalizers::BertNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::NFCNormalizer>())) {
    const auto& normalizer = py_obj.cast<const normalizers::NFCNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::NFKCNormalizer>())) {
    const auto& normalizer = py_obj.cast<const normalizers::NFKCNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::NFDNormalizer>())) {
    const auto& normalizer = py_obj.cast<const normalizers::NFDNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::NFKDNormalizer>())) {
    const auto& normalizer = py_obj.cast<const normalizers::NFKDNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::NmtNormalizer>())) {
    const auto& normalizer = py_obj.cast<const normalizers::NmtNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::ReplaceNormalizer>())) {
    const auto& normalizer =
        py_obj.cast<const normalizers::ReplaceNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::SequenceNormalizer>())) {
    const auto& normalizer =
        py_obj.cast<const normalizers::SequenceNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::StripAccentsNormalizer>())) {
    const auto& normalizer =
        py_obj.cast<const normalizers::StripAccentsNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::StripNormalizer>())) {
    const auto& normalizer = py_obj.cast<const normalizers::StripNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<normalizers::PrecompiledNormalizer>())) {
    const auto& normalizer =
        py_obj.cast<const normalizers::PrecompiledNormalizer&>();
    self->tokenizer.SetNormalizer(normalizer);
  } else if (py_obj.is(py::none())) {
    self->tokenizer.ReleaseNormaizer();
  } else {
    ret = 1;
    throw std::runtime_error("Need to assign the object of Normalizer");
  }
  return ret;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NEG
}

static PyObject* TokenizerPropertiesGetPreTokenizer(TokenizerObject* self,
                                                    void* closure) {
  py::object py_obj = py::cast(self->tokenizer.GetPreTokenizer());
  py_obj.inc_ref();
  return py_obj.ptr();
}

static int TokenizerPropertiesSetPreTokenizer(TokenizerObject* self,
                                              PyObject* value,
                                              void* closure) {
  TOKENIZERS_TRY
  py::handle py_obj(value);
  int ret = 0;
  if (pybind11::type::of(py_obj).is(
          py::type::of<pretokenizers::BertPreTokenizer>())) {
    const auto& pretokenizer =
        py_obj.cast<const pretokenizers::BertPreTokenizer&>();
    self->tokenizer.SetPreTokenizer(pretokenizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<pretokenizers::WhitespacePreTokenizer>())) {
    const auto& pretokenizer =
        py_obj.cast<const pretokenizers::WhitespacePreTokenizer&>();
    self->tokenizer.SetPreTokenizer(pretokenizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<pretokenizers::MetaSpacePreTokenizer>())) {
    const auto& pretokenizer =
        py_obj.cast<const pretokenizers::MetaSpacePreTokenizer&>();
    self->tokenizer.SetPreTokenizer(pretokenizer);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<pretokenizers::SequencePreTokenizer>())) {
    const auto& pretokenizer =
        py_obj.cast<const pretokenizers::SequencePreTokenizer&>();
    self->tokenizer.SetPreTokenizer(pretokenizer);
  } else if (py_obj.is(py::none())) {
    self->tokenizer.ReleasePreTokenizer();
  } else {
    ret = 1;
    throw std::runtime_error("Need to assign the object of PreTokenizer");
  }
  return ret;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NEG
}

static PyObject* TokenizerPropertiesGetModel(TokenizerObject* self,
                                             void* closure) {
  py::object py_obj = py::cast(self->tokenizer.GetModelPtr());
  py_obj.inc_ref();
  return py_obj.ptr();
}

static int TokenizerPropertiesSetModel(TokenizerObject* self,
                                       PyObject* value,
                                       void* closure) {
  TOKENIZERS_TRY
  py::handle py_obj(value);
  int ret = 0;
  if (pybind11::type::of(py_obj).is(py::type::of<models::WordPiece>())) {
    const auto& model = py_obj.cast<const models::WordPiece&>();
    self->tokenizer.SetModel(model);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<models::FasterWordPiece>())) {
    const auto& model = py_obj.cast<const models::FasterWordPiece&>();
    self->tokenizer.SetModel(model);
  } else if (pybind11::type::of(py_obj).is(py::type::of<models::BPE>())) {
    const auto& model = py_obj.cast<const models::BPE&>();
    self->tokenizer.SetModel(model);
  } else if (pybind11::type::of(py_obj).is(py::type::of<models::Unigram>())) {
    const auto& model = py_obj.cast<const models::Unigram&>();
    self->tokenizer.SetModel(model);
  } else {
    ret = 1;
    throw std::runtime_error("Need to assign the object of Model");
  }
  return ret;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NEG
}

static PyObject* TokenizerPropertiesGetPostProcessor(TokenizerObject* self,
                                                     void* closure) {
  py::object py_obj = py::cast(self->tokenizer.GetPostProcessorPtr());
  py_obj.inc_ref();
  return py_obj.ptr();
}

static int TokenizerPropertiesSetPostProcessor(TokenizerObject* self,
                                               PyObject* value,
                                               void* closure) {
  TOKENIZERS_TRY
  py::handle py_obj(value);
  int ret = 0;
  if (pybind11::type::of(py_obj).is(
          py::type::of<postprocessors::BertPostProcessor>())) {
    const auto& processor =
        py_obj.cast<const postprocessors::BertPostProcessor&>();
    self->tokenizer.SetPostProcessor(processor);
  } else if (pybind11::type::of(py_obj).is(
                 py::type::of<postprocessors::TemplatePostProcessor>())) {
    const auto& processor =
        py_obj.cast<const postprocessors::TemplatePostProcessor&>();
    self->tokenizer.SetPostProcessor(processor);
  } else if (py_obj.is(py::none())) {
    self->tokenizer.ReleasePostProcessor();
  } else {
    ret = 1;
    throw std::runtime_error("Need to assign the object of PostProcessor");
  }
  return ret;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NEG
}

static PyObject* TokenizerPropertiesGetPadding(TokenizerObject* self,
                                               void* closure) {
  if (!self->tokenizer.GetUsePadding()) {
    Py_RETURN_NONE;
  }
  auto pad_method = self->tokenizer.GetPadMethod();
  PyObject* py_dict = PyDict_New();
  PyDict_SetItem(py_dict, ToPyObject("pad_id"), ToPyObject(pad_method.pad_id_));
  PyDict_SetItem(py_dict,
                 ToPyObject("pad_token_type_id"),
                 ToPyObject(pad_method.pad_token_type_id_));
  PyDict_SetItem(
      py_dict, ToPyObject("pad_token"), ToPyObject(pad_method.pad_token_));
  if (pad_method.pad_to_multiple_of_ > 0) {
    PyDict_SetItem(py_dict,
                   ToPyObject("pad_to_multiple_of"),
                   ToPyObject(pad_method.pad_to_multiple_of_));
  } else {
    Py_INCREF(Py_None);
    PyDict_SetItem(py_dict, ToPyObject("pad_to_multiple_of"), Py_None);
  }

  PyDict_SetItem(
      py_dict,
      ToPyObject("direction"),
      ToPyObject((pad_method.direction_ == core::Direction::RIGHT) ? "right"
                                                                   : "left"));
  if (pad_method.strategy_ == core::PadStrategy::BATCH_LONGEST) {
    Py_INCREF(Py_None);
    PyDict_SetItem(py_dict, ToPyObject("length"), Py_None);
  } else {
    PyDict_SetItem(
        py_dict, ToPyObject("length"), ToPyObject(pad_method.pad_len_));
  }
  return py_dict;
}

static PyObject* TokenizerPropertiesGetTruncation(TokenizerObject* self,
                                                  void* closure) {
  if (!self->tokenizer.GetUseTruncation()) {
    Py_RETURN_NONE;
  }
  auto trunc_method = self->tokenizer.GetTruncMethod();
  PyObject* py_dict = PyDict_New();
  PyDict_SetItem(
      py_dict, ToPyObject("max_length"), ToPyObject(trunc_method.max_len_));
  PyDict_SetItem(
      py_dict, ToPyObject("stride"), ToPyObject(trunc_method.stride_));
  PyDict_SetItem(
      py_dict,
      ToPyObject("direction"),
      ToPyObject((trunc_method.direction_ == core::Direction::RIGHT) ? "right"
                                                                     : "left"));
  if (trunc_method.strategy_ == core::TruncStrategy::LONGEST_FIRST) {
    PyDict_SetItem(
        py_dict, ToPyObject("strategy"), ToPyObject("longest_first"));
  } else if (trunc_method.strategy_ == core::TruncStrategy::ONLY_FIRST) {
    PyDict_SetItem(py_dict, ToPyObject("strategy"), ToPyObject("only_first"));
  } else if (trunc_method.strategy_ == core::TruncStrategy::ONLY_SECOND) {
    PyDict_SetItem(py_dict, ToPyObject("strategy"), ToPyObject("only_second"));
  }
  return py_dict;
}

static PyObject* TokenizerPropertiesGetDecoder(TokenizerObject* self,
                                               void* closure) {
  py::object py_obj = py::cast(self->tokenizer.GetDecoderPtr());
  py_obj.inc_ref();
  return py_obj.ptr();
}

static int TokenizerPropertiesSetDecoder(TokenizerObject* self,
                                         PyObject* value,
                                         void* closure) {
  TOKENIZERS_TRY
  py::handle py_obj(value);
  int ret = 0;
  if (pybind11::type::of(py_obj).is(py::type::of<decoders::WordPiece>())) {
    const auto& decoder = py_obj.cast<const decoders::WordPiece&>();
    self->tokenizer.SetDecoder(decoder);
  } else if (py_obj.is(py::none())) {
    self->tokenizer.ReleaseDecoder();
  } else {
    ret = 1;
    throw std::runtime_error("Need to assign the object of Decoder");
  }
  return ret;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NEG
}

struct PyGetSetDef tokenizer_variable_properties[] = {
    {"normalizer",
     (getter)TokenizerPropertiesGetNormaizer,
     (setter)TokenizerPropertiesSetNormalizer,
     nullptr,
     nullptr},
    {"pretokenizer",
     (getter)TokenizerPropertiesGetPreTokenizer,
     (setter)TokenizerPropertiesSetPreTokenizer,
     nullptr,
     nullptr},
    {"model",
     (getter)TokenizerPropertiesGetModel,
     (setter)TokenizerPropertiesSetModel,
     nullptr,
     nullptr},
    {"postprocessor",
     (getter)TokenizerPropertiesGetPostProcessor,
     (setter)TokenizerPropertiesSetPostProcessor,
     nullptr,
     nullptr},
    {"padding",
     (getter)TokenizerPropertiesGetPadding,
     nullptr,
     nullptr,
     nullptr},
    {"truncation",
     (getter)TokenizerPropertiesGetTruncation,
     nullptr,
     nullptr,
     nullptr},
    {"decoder",
     (getter)TokenizerPropertiesGetDecoder,
     (setter)TokenizerPropertiesSetDecoder,
     nullptr,
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyObject* TokenizerNew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = reinterpret_cast<TokenizerObject*>(obj);
    new (&(v->tokenizer)) core::Tokenizer();
  }
  return obj;
}

static void TokenizerDealloc(TokenizerObject* self) {
  if (self->weakrefs != NULL)
    PyObject_ClearWeakRefs(reinterpret_cast<PyObject*>(self));
  self->tokenizer.~Tokenizer();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int TokenizerInit(PyObject* self, PyObject* args, PyObject* kwargs) {
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;
  // all kwargs
  PyObject* kw_model = NULL;
  // the keywords argument
  static char* kwlist[] = {const_cast<char*>("model"), NULL};
  // 'O' Store a Python object (without any conversion) in a C object pointer,
  // '|' Indicates that the remaining arguments in the Python argument list are
  // optional.
  bool flag_ =
      PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &kw_model);
  std::unordered_map<std::string, PyObject*> kws_map{{"model", kw_model}};

  auto py_tokenizer_ptr = reinterpret_cast<TokenizerObject*>(self);
  Py_ssize_t args_num = PyTuple_Size(args);

  if (args_num == 1) {
    py::object py_obj =
        py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(args, 0));
    if (pybind11::type::of(py_obj).is(py::type::of<models::WordPiece>())) {
      const auto& model = py_obj.cast<const models::WordPiece&>();
      py_tokenizer_ptr->tokenizer.SetModel(model);
    } else if (pybind11::type::of(py_obj).is(
                   py::type::of<models::FasterWordPiece>())) {
      const auto& model = py_obj.cast<const models::FasterWordPiece&>();
      py_tokenizer_ptr->tokenizer.SetModel(model);
    } else if (pybind11::type::of(py_obj).is(py::type::of<models::BPE>())) {
      const auto& model = py_obj.cast<const models::BPE&>();
      py_tokenizer_ptr->tokenizer.SetModel(model);
    } else if (pybind11::type::of(py_obj).is(py::type::of<models::Unigram>())) {
      const auto& model = py_obj.cast<const models::Unigram&>();
      py_tokenizer_ptr->tokenizer.SetModel(model);
    } else {
      std::ostringstream oss;
      oss << "Expected tpye of arguments is `model`";
      throw std::runtime_error(oss.str());
    }
    return 0;
  } else if (args_num >= 1) {
    std::ostringstream oss;
    oss << "Expected number of arguments is 0 or 1, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  return 1;
}

// def add_special_tokens(token)
static PyObject* AddSpecialTokens(TokenizerObject* self,
                                  PyObject* args,
                                  PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_special_tokens = NULL;
  static char* kwlist[] = {const_cast<char*>("tokens"), NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(
      args, kwargs, "|O", kwlist, &kw_special_tokens);
  Py_ssize_t args_num = PyTuple_Size(args);
  std::string tokens;
  if (args_num == (Py_ssize_t)1) {
    if (PyList_Check(kw_special_tokens)) {
      std::vector<core::AddedToken> added_tokens;
      Py_ssize_t tokens_num = PyList_GET_SIZE(kw_special_tokens);
      for (Py_ssize_t i = 0; i < tokens_num; ++i) {
        added_tokens.push_back(core::AddedToken(
            CastPyArg2AttrString(PyList_GetItem(kw_special_tokens, i), 0),
            true));
      }
      return ToPyObject(self->tokenizer.AddSpecialTokens(added_tokens));
    } else {
      // throw error
      throw std::runtime_error(
          "Need to pass the string list as to argument tokens");
    }
  } else {
    // throw error
    std::ostringstream oss;
    oss << "Expected number of arguments is 1, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def add_tokens(token)
static PyObject* AddTokens(TokenizerObject* self,
                           PyObject* args,
                           PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_tokens = NULL;
  static char* kwlist[] = {const_cast<char*>("tokens"), NULL};
  bool flag_ =
      PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &kw_tokens);
  Py_ssize_t args_num = PyTuple_Size(args);
  std::string tokens;
  if (args_num == (Py_ssize_t)1) {
    if (PyList_Check(kw_tokens)) {
      std::vector<core::AddedToken> added_tokens;
      Py_ssize_t tokens_num = PyList_GET_SIZE(kw_tokens);
      for (Py_ssize_t i = 0; i < tokens_num; ++i) {
        added_tokens.push_back(core::AddedToken(
            CastPyArg2AttrString(PyList_GetItem(kw_tokens, i), 0), true));
      }
      return ToPyObject(self->tokenizer.AddTokens(added_tokens));
    } else {
      throw std::runtime_error(
          "Need to pass the string list as to argument tokens");
    }
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is 1, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def enable_padding(direction="right", pad_id=0,
//                    pad_type_id=0, pad_token="[PAD]",
//                    length=None, ad_to_multiple_of=None)
static PyObject* EnablePadding(TokenizerObject* self,
                               PyObject* args,
                               PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_direction = NULL;
  PyObject* kw_pad_id = NULL;
  PyObject* kw_pad_type_id = NULL;
  PyObject* kw_pad_token = NULL;
  PyObject* kw_length = NULL;
  PyObject* kw_pad_to_multiple_of = NULL;
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;
  static char* kwlist[] = {const_cast<char*>("direction"),
                           const_cast<char*>("pad_id"),
                           const_cast<char*>("pad_type_id"),
                           const_cast<char*>("pad_token"),
                           const_cast<char*>("length"),
                           const_cast<char*>("pad_to_multiple_of"),
                           NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(args,
                                           kwargs,
                                           "|OOOOOO",
                                           kwlist,
                                           &kw_direction,
                                           &kw_pad_id,
                                           &kw_pad_type_id,
                                           &kw_pad_token,
                                           &kw_length,
                                           &kw_pad_to_multiple_of);
  Py_ssize_t args_num = PyTuple_Size(args);
  std::string direction = "right";
  uint32_t pad_id = 0;
  uint32_t pad_type_id = 0;
  std::string pad_token = "[PAD]";
  uint32_t* length_ptr = nullptr;
  uint32_t* pad_to_multiple_of_ptr = nullptr;
  uint32_t length = 0;
  uint32_t pad_to_multiple_of = 0;
  VLOG(6) << "args_num: " << args_num << ", flag_kwargs: " << flag_kwargs;
  VLOG(6) << "kw_direction: " << kw_direction;
  VLOG(6) << "kw_pad_id: " << kw_pad_id;
  VLOG(6) << "kw_pad_type_id: " << kw_pad_type_id;
  VLOG(6) << "kw_pad_token: " << kw_pad_token;
  VLOG(6) << "kw_length: " << kw_length;
  VLOG(6) << "kw_pad_to_multiple_of: " << kw_pad_to_multiple_of;
  if (args_num >= (Py_ssize_t)0 && args_num <= (Py_ssize_t)6) {
    if ((args_num == 0 && flag_kwargs && kw_direction) || (args_num >= 1)) {
      direction = CastPyArg2AttrString(kw_direction, 0);
    }
    if ((args_num <= 1 && flag_kwargs && kw_pad_id) || (args_num >= 2)) {
      pad_id = CastPyArg2AttrSize_t(kw_pad_id, 1);
    }
    if ((args_num <= 2 && flag_kwargs && kw_pad_type_id) || (args_num >= 3)) {
      pad_type_id = CastPyArg2AttrSize_t(kw_pad_type_id, 2);
    }
    if ((args_num <= 3 && flag_kwargs && kw_pad_token) || (args_num >= 4)) {
      pad_token = CastPyArg2AttrString(kw_pad_token, 3);
    }
    if ((args_num <= 4 && flag_kwargs && kw_length) || (args_num >= 5)) {
      if (!(kw_length == Py_None)) {
        length = CastPyArg2AttrSize_t(kw_length, 4);
        length_ptr = &length;
      }
    }
    if ((args_num <= 5 && flag_kwargs && kw_pad_to_multiple_of) ||
        (args_num == 6)) {
      if (!(kw_pad_to_multiple_of == Py_None)) {
        pad_to_multiple_of = CastPyArg2AttrSize_t(kw_pad_to_multiple_of, 5);
        pad_to_multiple_of_ptr = &pad_to_multiple_of;
      }
    }
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 0 to 6, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  core::Direction pad_direction;
  if (direction == "right") {
    pad_direction = core::Direction::RIGHT;
  } else if (direction == "left") {
    pad_direction = core::Direction::LEFT;
  } else {
    throw std::runtime_error(
        "The direction args should be \"right\" or \"left\"");
  }
  self->tokenizer.EnablePadMethod(pad_direction,
                                  pad_id,
                                  pad_type_id,
                                  pad_token,
                                  length_ptr,
                                  pad_to_multiple_of_ptr);
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def disable_padding()
static PyObject* DisablePadding(TokenizerObject* self,
                                PyObject* args,
                                PyObject* kwargs) {
  TOKENIZERS_TRY
  Py_ssize_t args_num = PyTuple_Size(args);
  if (args_num == (Py_ssize_t)0) {
    self->tokenizer.DisablePadMethod();
    Py_RETURN_NONE;
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is 0, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def enable_truncation(max_length, stride=0, strategy="longest_first",
// direction="right")
static PyObject* EnableTruncation(TokenizerObject* self,
                                  PyObject* args,
                                  PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_max_length = NULL;
  PyObject* kw_stride = NULL;
  PyObject* kw_strategy = NULL;
  PyObject* kw_direction = NULL;
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;
  static char* kwlist[] = {const_cast<char*>("max_length"),
                           const_cast<char*>("stride"),
                           const_cast<char*>("strategy"),
                           const_cast<char*>("direction"),
                           NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(args,
                                           kwargs,
                                           "|OOOO",
                                           kwlist,
                                           &kw_max_length,
                                           &kw_stride,
                                           &kw_strategy,
                                           &kw_direction);
  Py_ssize_t args_num = PyTuple_Size(args);
  uint32_t max_length = 0;
  uint32_t stride = 0;
  std::string strategy = "longest_first";
  std::string direction = "right";

  if (args_num >= (Py_ssize_t)0 && args_num <= (Py_ssize_t)4) {
    max_length = CastPyArg2AttrSize_t(kw_max_length, 0);
    if ((args_num <= 1 && flag_kwargs && kw_stride) || (args_num >= 2)) {
      stride = CastPyArg2AttrSize_t(kw_stride, 1);
    }
    if ((args_num <= 2 && flag_kwargs && kw_strategy) || (args_num >= 3)) {
      strategy = CastPyArg2AttrString(kw_strategy, 2);
    }
    if ((args_num <= 3 && flag_kwargs && kw_direction) || (args_num >= 4)) {
      direction = CastPyArg2AttrString(kw_direction, 3);
    }
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments 1 to 4, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }

  core::TruncStrategy trunc_strategy;
  if (strategy == "longest_first") {
    trunc_strategy = core::TruncStrategy::LONGEST_FIRST;
  } else if (strategy == "only_first") {
    trunc_strategy = core::TruncStrategy::ONLY_FIRST;
  } else if (strategy == "only_second") {
    trunc_strategy = core::TruncStrategy::ONLY_SECOND;
  } else {
    throw std::runtime_error(
        "The strategy args should be \"longest_first\", \"only_first\" or "
        "\"only_second\"");
  }
  core::Direction trunc_direction;
  if (direction == "right") {
    trunc_direction = core::Direction::RIGHT;
  } else if (direction == "left") {
    trunc_direction = core::Direction::LEFT;
  } else {
    throw std::runtime_error(
        "The direction args should be \"right\" or \"left\"");
  }
  self->tokenizer.EnableTruncMethod(
      max_length, stride, trunc_direction, trunc_strategy);
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def disable_truncation()
static PyObject* DisableTruncation(TokenizerObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  TOKENIZERS_TRY
  Py_ssize_t args_num = PyTuple_Size(args);
  if (args_num == (Py_ssize_t)0) {
    self->tokenizer.DisableTruncMethod();
    Py_RETURN_NONE;
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is 0, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def get_vocab(with_added_vocabulary=True)
static PyObject* GetVocab(TokenizerObject* self,
                          PyObject* args,
                          PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_with_added_vocabulary = NULL;
  static char* kwlist[] = {const_cast<char*>("with_added_vocabulary"), NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(
      args, kwargs, "|O", kwlist, &kw_with_added_vocabulary);
  Py_ssize_t args_num = PyTuple_Size(args);
  bool with_added_vocabulary = true;
  if (args_num == (Py_ssize_t)0) {
    with_added_vocabulary = true;
    py::object py_obj =
        py::cast(self->tokenizer.GetVocab(with_added_vocabulary));
    py_obj.inc_ref();
    return py_obj.ptr();
  } else if (args_num == (Py_ssize_t)1) {
    if (PyBool_Check(kw_with_added_vocabulary)) {
      with_added_vocabulary =
          CastPyArg2AttrBoolean(kw_with_added_vocabulary, 0);
      py::object py_obj =
          py::cast(self->tokenizer.GetVocab(with_added_vocabulary));
      py_obj.inc_ref();
      return py_obj.ptr();
    } else {
      // throw error
    }
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 0 to 1, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def get_vocab_size(with_added_vocabulary=True)
static PyObject* GetVocabSize(TokenizerObject* self,
                              PyObject* args,
                              PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_with_added_vocabulary = NULL;
  static char* kwlist[] = {const_cast<char*>("with_added_vocabulary"), NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(
      args, kwargs, "|O", kwlist, &kw_with_added_vocabulary);
  Py_ssize_t args_num = PyTuple_Size(args);
  bool with_added_vocabulary = true;
  if (args_num == (Py_ssize_t)0) {
    with_added_vocabulary = true;
    return ToPyObject(self->tokenizer.GetVocabSize(with_added_vocabulary));
  } else if (args_num == (Py_ssize_t)1) {
    if (PyBool_Check(kw_with_added_vocabulary)) {
      with_added_vocabulary =
          CastPyArg2AttrBoolean(kw_with_added_vocabulary, 0);
      return ToPyObject(self->tokenizer.GetVocabSize(with_added_vocabulary));
    } else {
      // throw error
    }
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is 0, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def encode(sequence, pair=None, is_pretokenized=False,
// add_special_tokens=True)
static PyObject* Encode(TokenizerObject* self,
                        PyObject* args,
                        PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_sequence = NULL;
  PyObject* kw_pair = NULL;
  PyObject* kw_is_pretokenized = NULL;
  PyObject* kw_add_special_tokens = NULL;
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;
  static char* kwlist[] = {const_cast<char*>("sequence"),
                           const_cast<char*>("pair"),
                           const_cast<char*>("is_pretokenized"),
                           const_cast<char*>("add_special_tokens"),
                           NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(args,
                                           kwargs,
                                           "|OOOO",
                                           kwlist,
                                           &kw_sequence,
                                           &kw_pair,
                                           &kw_is_pretokenized,
                                           &kw_add_special_tokens);
  Py_ssize_t args_num = PyTuple_Size(args);
  if (args_num >= (Py_ssize_t)1 && args_num <= (Py_ssize_t)4) {
    bool is_pretokenized = false;
    bool add_special_tokens = true;
    bool has_pair = false;
    core::Encoding encoding;
    core::Encoding pair_encoding;
    core::Encoding result_encoding;

    if ((args_num <= 2 && flag_kwargs && kw_is_pretokenized) ||
        (args_num >= 3)) {
      is_pretokenized = CastPyArg2AttrBoolean(kw_is_pretokenized, 2);
    }
    if ((args_num <= 3 && flag_kwargs && kw_add_special_tokens) ||
        (args_num >= 4)) {
      add_special_tokens = CastPyArg2AttrBoolean(kw_add_special_tokens, 3);
    }
    if (is_pretokenized) {
      if (PyList_Check(kw_sequence) || PyTuple_Check(kw_sequence)) {
        std::vector<std::string> sequence_array =
            CastPyArg2VectorOfStr(kw_sequence, 0);
        std::vector<std::string> pair_array;
        if ((args_num <= 1 && flag_kwargs && kw_pair && kw_pair != Py_None) ||
            (args_num >= 2)) {
          has_pair = true;
          pair_array = CastPyArg2VectorOfStr(kw_pair, 1);
        }
        self->tokenizer.EncodeSingleString(
            sequence_array, 0, core::OffsetType::CHAR, &encoding);
        core::Encoding* pair_encoding_ptr = nullptr;
        if (has_pair) {
          self->tokenizer.EncodeSingleString(
              pair_array, 0, core::OffsetType::CHAR, &pair_encoding);
          pair_encoding_ptr = &pair_encoding;
        }
        self->tokenizer.PostProcess(
            &encoding, pair_encoding_ptr, add_special_tokens, &result_encoding);
      } else {
        // throw error
        std::ostringstream oss;
        oss << "The sequence should be list of string when "
               "is_pretokenized=True";
        throw std::runtime_error(oss.str());
      }
    } else {
      std::string sequence = CastPyArg2AttrString(kw_sequence, 0);
      std::string pair;
      if (((args_num <= 1 && flag_kwargs && kw_pair) || (args_num >= 2)) &&
          kw_pair != Py_None) {
        has_pair = true;
        pair = CastPyArg2AttrString(kw_pair, 1);
      }
      self->tokenizer.EncodeSingleString(
          sequence, 0, core::OffsetType::CHAR, &encoding);
      core::Encoding* pair_encoding_ptr = nullptr;
      if (has_pair) {
        self->tokenizer.EncodeSingleString(
            pair, 1, core::OffsetType::CHAR, &pair_encoding);
        pair_encoding_ptr = &pair_encoding;
      }
      self->tokenizer.PostProcess(
          &encoding, pair_encoding_ptr, add_special_tokens, &result_encoding);
    }
    py::object py_obj = py::cast(result_encoding);
    py_obj.inc_ref();
    return py_obj.ptr();
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 1 to 4, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def encode(input, add_special_tokens=True, is_pretokenized=False)
static PyObject* EncodeBatch(TokenizerObject* self,
                             PyObject* args,
                             PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_input = NULL;
  PyObject* kw_special_tokens = NULL;
  PyObject* kw_is_pretokenized = NULL;
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;
  static char* kwlist[] = {const_cast<char*>("input"),
                           const_cast<char*>("add_special_tokens"),
                           const_cast<char*>("is_pretokenized"),
                           NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(args,
                                           kwargs,
                                           "|OOO",
                                           kwlist,
                                           &kw_input,
                                           &kw_special_tokens,
                                           &kw_is_pretokenized);
  bool add_special_tokens = true;
  bool is_pretokenized = false;
  Py_ssize_t args_num = PyTuple_Size(args);
  VLOG(6) << " args_num: " << args_num << ", flag_kwargs: " << flag_kwargs
          << ", flag_: " << flag_;
  std::vector<core::EncodeInput> batch_encode_input;
  if (args_num >= (Py_ssize_t)1 && args_num <= (Py_ssize_t)3) {
    if ((args_num <= 1 && flag_kwargs && kw_special_tokens) ||
        (args_num >= 2)) {
      add_special_tokens = CastPyArg2AttrBoolean(kw_special_tokens, 1);
    }
    if ((args_num <= 2 && kw_is_pretokenized && flag_kwargs) || args_num == 3) {
      is_pretokenized = CastPyArg2AttrBoolean(kw_is_pretokenized, 2);
    }
    if (PyList_Check(kw_input)) {
      Py_ssize_t list_size = PyList_Size(kw_input);
      for (Py_ssize_t i = 0; i < list_size; ++i) {
        PyObject* item = PyList_GetItem(kw_input, i);
        // Has pair
        if (PyTuple_Check(item) && PyTuple_Size(item) == 2) {
          PyObject* text = PyTuple_GetItem(item, 0);
          PyObject* text_pair = PyTuple_GetItem(item, 1);
          // pretokenized
          if (is_pretokenized) {
            Py_ssize_t pretokenized_size = PyList_Size(item);
            std::vector<std::string> text_vec;
            std::vector<std::string> text_pair_vec;
            for (Py_ssize_t j = 0; j < pretokenized_size; ++j) {
              PyObject* py_text = PyList_GetItem(text, j);
              PyObject* py_text_pair = PyList_GetItem(text_pair, j);
              text_vec.push_back(CastPyArg2AttrString(py_text, 0));
              text_pair_vec.push_back(CastPyArg2AttrString(py_text_pair, 1));
            }
            batch_encode_input.push_back(
                std::pair<core::InputString, core::InputString>{text_vec,
                                                                text_pair_vec});
          } else {
            batch_encode_input.push_back(
                std::pair<core::InputString, core::InputString>{
                    CastPyArg2AttrString(text, 0),
                    CastPyArg2AttrString(text_pair, 1)});
          }
        } else {
          // Only get text
          if (is_pretokenized) {
            Py_ssize_t pretokenized_size = PyList_Size(item);
            std::vector<std::string> str_vec(pretokenized_size);
            for (Py_ssize_t j = 0; j < pretokenized_size; ++j) {
              PyObject* py_text = PyList_GetItem(item, j);
              str_vec[j] = CastPyArg2AttrString(py_text, 0);
            }
            batch_encode_input.push_back(str_vec);
          } else {
            batch_encode_input.push_back(CastPyArg2AttrString(item, 0));
          }
        }
      }
    } else {
      std::ostringstream oss;
      oss << "Expected the type of input argument is list";
      throw std::runtime_error(oss.str());
    }
    std::vector<core::Encoding> result_encodings;
    self->tokenizer.EncodeBatchStrings(
        batch_encode_input, &result_encodings, add_special_tokens);
    py::object py_obj = py::cast(result_encodings);
    py_obj.inc_ref();
    return py_obj.ptr();
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 1 to 2, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def id_to_token(id)
static PyObject* IdToToken(TokenizerObject* self,
                           PyObject* args,
                           PyObject* kwargs) {
  PyObject* kw_id = NULL;
  static char* kwlist[] = {const_cast<char*>("id"), NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &kw_id);
  Py_ssize_t args_num = PyTuple_Size(args);
  if (args_num == (Py_ssize_t)1) {
    if (PyLong_Check(kw_id)) {
      uint32_t id = PyLong_AsLong(kw_id);
      std::string token;
      if (self->tokenizer.IdToToken(id, &token)) {
        return ToPyObject(token);
      }
      Py_RETURN_NONE;
    } else {
      // throw error
    }
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is 1, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
}

// def token_to_id(token)
static PyObject* TokenToId(TokenizerObject* self,
                           PyObject* args,
                           PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_token = NULL;
  static char* kwlist[] = {const_cast<char*>("token"), NULL};
  bool flag_ =
      PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &kw_token);
  Py_ssize_t args_num = PyTuple_Size(args);
  std::string token = "";
  if (args_num == (Py_ssize_t)1) {
    token = CastPyArg2AttrString(kw_token, 0);
    uint32_t id;
    if (self->tokenizer.TokenToId(token, &id)) {
      return ToPyObject(id);
    }
    Py_RETURN_NONE;
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is 1, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def num_special_tokens_to_add(is_pair)
static PyObject* NumSpecialTokensToAdd(TokenizerObject* self,
                                       PyObject* args,
                                       PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_is_pair = NULL;
  static char* kwlist[] = {const_cast<char*>("is_pair"), NULL};
  bool flag_ =
      PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &kw_is_pair);
  Py_ssize_t args_num = PyTuple_Size(args);
  bool is_pair = false;
  if (args_num == (Py_ssize_t)1) {
    is_pair = CastPyArg2AttrBoolean(kw_is_pair, 0);
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is 1, but recive " << args_num;
    throw std::runtime_error(oss.str());
  }
  auto postprocessor_ptr = self->tokenizer.GetPostProcessorPtr();
  if (postprocessor_ptr == nullptr) {
    return ToPyObject(0);
  }
  return ToPyObject(postprocessor_ptr->AddedTokensNum(is_pair));
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}


static PyObject* Save(TokenizerObject* self, PyObject* args, PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_path = NULL;
  PyObject* kw_pretty = NULL;
  static char* kwlist[] = {
      const_cast<char*>("path"), const_cast<char*>("pretty"), NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(
      args, kwargs, "|OO", kwlist, &kw_path, &kw_pretty);
  bool pretty = true;
  Py_ssize_t args_num = PyTuple_Size(args);
  if (args_num >= (Py_ssize_t)1 && args_num <= (Py_ssize_t)2) {
    if (args_num == (Py_ssize_t)2) {
      pretty = CastPyArg2AttrBoolean(kw_pretty, 1);
    }
    std::string path = CastPyArg2AttrString(kw_path, 0);
    self->tokenizer.Save(path, pretty);
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 1 to 2, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  Py_RETURN_NONE;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* ToStr(TokenizerObject* self,
                       PyObject* args,
                       PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_pretty = NULL;
  static char* kwlist[] = {const_cast<char*>("pretty"), NULL};
  bool flag_ =
      PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &kw_pretty);
  bool pretty = true;
  Py_ssize_t args_num = PyTuple_Size(args);
  std::string json_str;
  if (args_num >= (Py_ssize_t)0 && args_num <= (Py_ssize_t)1) {
    if (args_num == (Py_ssize_t)1) {
      pretty = CastPyArg2AttrBoolean(kw_pretty, 0);
    }
    self->tokenizer.ToJsonStr(&json_str, pretty);
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 1 to 2, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  return ToPyObject(json_str);
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* FromStr(TokenizerObject* self,
                         PyObject* args,
                         PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_json = NULL;
  static char* kwlist[] = {const_cast<char*>("json"), NULL};
  bool flag_ =
      PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &kw_json);
  Py_ssize_t args_num = PyTuple_Size(args);
  std::string json_str;
  core::Tokenizer tokenizer;
  if (args_num == (Py_ssize_t)1) {
    json_str = CastPyArg2AttrString(kw_json, 0);
    tokenizer = core::Tokenizer::LoadFromStr(json_str);
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 1 to 2, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  TokenizerObject* obj =
      (TokenizerObject*)TokenizerNew(p_tokenizer_type, NULL, NULL);
  obj->tokenizer = tokenizer;
  return (PyObject*)obj;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* FromFile(TokenizerObject* self,
                          PyObject* args,
                          PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_path = NULL;
  static char* kwlist[] = {const_cast<char*>("json"), NULL};
  bool flag_ =
      PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &kw_path);
  Py_ssize_t args_num = PyTuple_Size(args);
  std::string path;
  core::Tokenizer tokenizer;
  if (args_num == (Py_ssize_t)1) {
    path = CastPyArg2AttrString(kw_path, 0);
    tokenizer = core::Tokenizer::LoadFromFile(path);
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 1 to 2, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  TokenizerObject* obj =
      (TokenizerObject*)TokenizerNew(p_tokenizer_type, NULL, NULL);
  obj->tokenizer = tokenizer;
  return (PyObject*)obj;
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def decode(self, ids, skip_special_tokens=True):
static PyObject* Decode(TokenizerObject* self,
                        PyObject* args,
                        PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_ids = NULL;
  PyObject* kw_skip_special_tokens = NULL;
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;
  static char* kwlist[] = {
      const_cast<char*>("ids"), const_cast<char*>("skip_special_tokens"), NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(
      args, kwargs, "|OO", kwlist, &kw_ids, &kw_skip_special_tokens);
  bool skip_special_tokens = true;
  Py_ssize_t args_num = PyTuple_Size(args);
  VLOG(6) << " args_num: " << args_num << ", flag_kwargs: " << flag_kwargs
          << ", flag_: " << flag_;
  if (args_num >= (Py_ssize_t)1 && args_num <= (Py_ssize_t)2) {
    if (args_num == (Py_ssize_t)2 || (flag_kwargs && kw_skip_special_tokens)) {
      skip_special_tokens = CastPyArg2AttrBoolean(kw_skip_special_tokens, 1);
    }
    auto ids = CastPyArg2VectorOfInt<uint32_t>(kw_ids, 0);
    std::string result;
    self->tokenizer.Decode(ids, &result, skip_special_tokens);
    return ToPyObject(result);
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 1 to 2, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

// def decode_batch(self, sequences, skip_special_tokens=True):
static PyObject* DecodeBatch(TokenizerObject* self,
                             PyObject* args,
                             PyObject* kwargs) {
  TOKENIZERS_TRY
  PyObject* kw_sequences = NULL;
  PyObject* kw_skip_special_tokens = NULL;
  bool flag_kwargs = false;
  if (kwargs) flag_kwargs = true;
  static char* kwlist[] = {const_cast<char*>("sequences"),
                           const_cast<char*>("skip_special_tokens"),
                           NULL};
  bool flag_ = PyArg_ParseTupleAndKeywords(
      args, kwargs, "|OO", kwlist, &kw_sequences, &kw_skip_special_tokens);
  bool skip_special_tokens = true;
  Py_ssize_t args_num = PyTuple_Size(args);
  VLOG(6) << " args_num: " << args_num << ", flag_kwargs: " << flag_kwargs
          << ", flag_: " << flag_;
  if (args_num >= (Py_ssize_t)1 && args_num <= (Py_ssize_t)2) {
    if (args_num == (Py_ssize_t)2 || (flag_kwargs && kw_skip_special_tokens)) {
      skip_special_tokens = CastPyArg2AttrBoolean(kw_skip_special_tokens, 1);
    }
    std::vector<std::vector<uint32_t>> batch_ids;
    PyObject* item = nullptr;
    if (PyTuple_Check(kw_sequences)) {
      Py_ssize_t len = PyTuple_Size(kw_sequences);
      for (Py_ssize_t i = 0; i < len; i++) {
        item = PyTuple_GetItem(kw_sequences, i);
        batch_ids.emplace_back(CastPyArg2VectorOfInt<uint32_t>(item, 0));
      }
    } else if (PyList_Check(kw_sequences)) {
      Py_ssize_t len = PyList_Size(kw_sequences);
      for (Py_ssize_t i = 0; i < len; i++) {
        item = PyList_GetItem(kw_sequences, i);
        batch_ids.emplace_back(CastPyArg2VectorOfInt<uint32_t>(item, 0));
      }
    } else {
      std::ostringstream oss;
      oss << "Args sequences need to be int of list or tuple";
      throw std::runtime_error(oss.str());
    }
    std::vector<std::string> result;
    self->tokenizer.DecodeBatch(batch_ids, &result, skip_special_tokens);
    return ToPyObject(result);
  } else {
    std::ostringstream oss;
    oss << "Expected number of arguments is from 1 to 2, but recive "
        << args_num;
    throw std::runtime_error(oss.str());
  }
  TOKENIZERS_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef tokenizer_variable_methods[] = {
    {"add_special_tokens",
     (PyCFunction)(void (*)(void))AddSpecialTokens,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"add_tokens",
     (PyCFunction)(void (*)(void))AddTokens,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"enable_padding",
     (PyCFunction)(void (*)(void))EnablePadding,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"disable_padding",
     (PyCFunction)(void (*)(void))DisablePadding,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"enable_truncation",
     (PyCFunction)(void (*)(void))EnableTruncation,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"disable_truncation",
     (PyCFunction)(void (*)(void))DisableTruncation,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"get_vocab",
     (PyCFunction)(void (*)(void))GetVocab,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"get_vocab_size",
     (PyCFunction)(void (*)(void))GetVocabSize,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"encode",
     (PyCFunction)(void (*)(void))Encode,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"encode_batch",
     (PyCFunction)(void (*)(void))EncodeBatch,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"decode",
     (PyCFunction)(void (*)(void))Decode,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"decode_batch",
     (PyCFunction)(void (*)(void))DecodeBatch,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"id_to_token",
     (PyCFunction)(void (*)(void))IdToToken,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"token_to_id",
     (PyCFunction)(void (*)(void))TokenToId,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"num_special_tokens_to_add",
     (PyCFunction)(void (*)(void))NumSpecialTokensToAdd,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"save",
     (PyCFunction)(void (*)(void))Save,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"to_str",
     (PyCFunction)(void (*)(void))ToStr,
     METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"from_str",
     (PyCFunction)(void (*)(void))FromStr,
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     NULL},
    {"from_file",
     (PyCFunction)(void (*)(void))FromFile,
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     NULL},
    // TODO(zhoushunjie): Need to implement
    // {"from_buffer",
    //  (PyCFunction)(void (*)(void))NumSpecialTokensToAdd,
    //  METH_VARARGS | METH_KEYWORDS | METH_STATIC,
    //  NULL},
    // {"from_pretrained",
    //  (PyCFunction)(void (*)(void))NumSpecialTokensToAdd,
    //  METH_VARARGS | METH_KEYWORDS | METH_STATIC,
    //  NULL},
    {NULL, NULL, 0, NULL}};

void BindTokenizers(pybind11::module* m) {
  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  heap_type->ht_name = ToPyObject("Tokenizer");
  heap_type->ht_qualname = ToPyObject("Tokenizer");
  auto type = &heap_type->ht_type;
  type->tp_name = "Tokenizer";
  type->tp_basicsize = sizeof(TokenizerObject);
  type->tp_dealloc = (destructor)TokenizerDealloc;
  type->tp_as_number = &number_methods;
  type->tp_as_sequence = &sequence_methods;
  type->tp_as_mapping = &mapping_methods;
  type->tp_methods = tokenizer_variable_methods;
  type->tp_getset = tokenizer_variable_properties;
  type->tp_init = TokenizerInit;
  type->tp_new = TokenizerNew;
  Py_INCREF(&PyBaseObject_Type);
  type->tp_base = reinterpret_cast<PyTypeObject*>(&PyBaseObject_Type);
  type->tp_flags |=
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
#if PY_VERSION_HEX >= 0x03050000
  type->tp_as_async = &heap_type->as_async;
#endif
  p_tokenizer_type = type;

  if (PyType_Ready(type) < 0) {
    throw "Init Tokenizers error in BindTokenizers(PyType_Ready).";
    return;
  }

  Py_INCREF(type);
  if (PyModule_AddObject(
          m->ptr(), "Tokenizer", reinterpret_cast<PyObject*>(type)) < 0) {
    Py_DECREF(type);
    Py_DECREF(m->ptr());
    throw "Init Tokenizers error in BindTokenizers(PyModule_AddObject).";
    return;
  }
}

}  // namespace pybind
}  // namespace faster_tokenizer
}  // namespace paddlenlp
