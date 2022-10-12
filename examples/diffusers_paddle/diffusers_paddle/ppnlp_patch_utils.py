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

import functools, builtins, os, json, copy, numpy as np
from .utils import logging, is_paddle_available, is_paddlenlp_available
from types import FunctionType, MethodType
from typing import Any, Dict, Tuple, Union


def copy_func(f):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)"
    if not isinstance(f, FunctionType): return copy.copy(f)
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__,
                      f.__closure__)
    fn.__kwdefaults__ = f.__kwdefaults__
    fn.__dict__.update(f.__dict__)
    fn.__annotations__.update(f.__annotations__)
    fn.__qualname__ = f.__qualname__
    return fn


# copied from https://github.com/fastai/fastcore/blob/c9b4c088d3706569c076e7c197c724730be190ab/fastcore/basics.py#L938-L954
def patch_to(cls, as_prop=False, cls_method=False):
    "Decorator: add `f` to `cls`"
    if not isinstance(cls, (tuple, list)): cls = (cls, )

    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            nm = f.__name__
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS:
                setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{nm}"
            if cls_method:
                setattr(c_, nm, MethodType(nf, c_))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))

    return _inner


if is_paddle_available() and is_paddlenlp_available():
    from paddlenlp.transformers.clip.feature_extraction import CLIPFeatureExtractor
    from paddlenlp.transformers import CLIPTextModel, PretrainedModel

    @patch_to(CLIPTextModel)
    def get_input_embeddings(self):
        return self.text_model.token_embedding

    @patch_to(CLIPTextModel)
    def set_input_embeddings(self, value):
        self.text_model.token_embedding = value

    @patch_to(CLIPTextModel)
    def get_model_config(self):

        def get_config(model):
            model_config = model.init_config
            for key, value in model_config.items():
                if key == "init_args":
                    args = []
                    for arg in value:
                        args.append(
                            get_config(arg) if isinstance(arg, PretrainedModel
                                                          ) else arg)
                    model_config[key] = tuple(args)
                elif isinstance(value, PretrainedModel):
                    model_config[key] = value.init_config
            return model_config

        model_config = get_config(self)
        model_config['vocab_size'] = self.base_model.config['vocab_size']
        return model_config

    logger = logging.get_logger(__name__)

    @patch_to(CLIPFeatureExtractor)
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @patch_to(CLIPFeatureExtractor)
    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        return output

    @patch_to(CLIPFeatureExtractor)
    def to_json_string(self) -> str:
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    @patch_to(CLIPFeatureExtractor, cls_method=True)
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str,
                                                                  os.PathLike],
                        **kwargs):
        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(
            pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(feature_extractor_dict, **kwargs)

    @patch_to(CLIPFeatureExtractor)
    def save_pretrained(self, save_directory: Union[str, os.PathLike],
                        **kwargs):
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        # # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # # loaded from the Hub.
        # if self._auto_class is not None:
        #     custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_feature_extractor_file = os.path.join(save_directory,
                                                     FEATURE_EXTRACTOR_NAME)

        self.to_json_file(output_feature_extractor_file)
        logger.info(
            f"Feature extractor saved in {output_feature_extractor_file}")

        return [output_feature_extractor_file]

    @patch_to(CLIPFeatureExtractor, cls_method=True)
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs):
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        feature_extractor = cls(**feature_extractor_dict)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(feature_extractor, key):
                setattr(feature_extractor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor

    ################################################get_feature_extractor_dict
    ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
    _is_offline_mode = True if os.environ.get(
        "TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False
    FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"

    def is_offline_mode():
        return _is_offline_mode

    @patch_to(CLIPFeatureExtractor, cls_method=True)
    def get_feature_extractor_dict(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.
        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
        """
        # cache_dir = kwargs.pop("cache_dir", None)
        # force_download = kwargs.pop("force_download", False)
        # resume_download = kwargs.pop("resume_download", False)
        # proxies = kwargs.pop("proxies", None)
        # use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        # revision = kwargs.pop("revision", None)

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {
            "file_type": "feature extractor",
            "from_auto_class": from_auto_class
        }
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            feature_extractor_file = os.path.join(pretrained_model_name_or_path,
                                                  FEATURE_EXTRACTOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_feature_extractor_file = pretrained_model_name_or_path
            is_local = True
        else:
            feature_extractor_file = FEATURE_EXTRACTOR_NAME
            try:
                resolved_feature_extractor_file = os.path.join(
                    pretrained_model_name_or_path, feature_extractor_file)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load feature extractor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {FEATURE_EXTRACTOR_NAME} file")

        try:
            # Load feature_extractor dict
            with open(resolved_feature_extractor_file, "r",
                      encoding="utf-8") as reader:
                text = reader.read()
            feature_extractor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_feature_extractor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(
                f"loading configuration file {resolved_feature_extractor_file}")
        else:
            logger.info(
                f"loading configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}"
            )

        return feature_extractor_dict, kwargs

    import paddle

    @patch_to(paddle.Tensor)
    def repeat_interleave(self, tensor, axis=0):
        if self.dtype == paddle.float16:
            return paddle.repeat_interleave(self.astype("float32"),
                                            tensor,
                                            axis=axis).astype(self.dtype)
        else:
            return paddle.repeat_interleave(self, tensor, axis=axis)

    @patch_to(PretrainedModel, as_prop=True)
    def dtype(self):
        try:
            return next(self.named_parameters())[1].dtype
        except StopIteration:
            return paddle.get_default_dtype()

    @patch_to(PretrainedModel, as_prop=True)
    def device(self):
        try:
            return next(self.named_parameters())[1].place
        except StopIteration:
            return paddle.get_device()
