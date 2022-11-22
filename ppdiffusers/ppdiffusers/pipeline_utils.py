# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn

import PIL
from packaging import version
from PIL import Image
from tqdm.auto import tqdm

from .configuration_utils import ConfigMixin
from .schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from .utils import (
    CONFIG_NAME,
    PPDIFFUSERS_CACHE,
    ONNX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    BaseOutput,
    deprecate,
    is_paddle_available,
    is_paddlenlp_available,
    logging,
)

from . import OnnxRuntimeModel

if is_paddlenlp_available():
    from paddlenlp.transformers import PretrainedModel, PretrainedTokenizer, FeatureExtractionMixin, ProcessorMixin

INDEX_FILE = "model_state.pdparams"
CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
DUMMY_MODULES_FOLDER = "ppdiffusers.utils"
PADDLENLP_DUMMY_MODULES_FOLDER = "paddlenlp.transformers.utils"

logger = logging.get_logger(__name__)

LOADABLE_CLASSES = {
    "ppdiffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "paddlenlp.transformers": {
        "PretrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PretrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
    },
    "onnxruntime.training": {
        "ORTModule": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised samples of shape `(batch_size, num_channels, sample_rate)`. Numpy array present the
            denoised audio samples of the diffusion pipeline.
    """

    audios: np.ndarray


class DiffusionPipeline(ConfigMixin):
    r"""
    Base class for all models.

    [`DiffusionPipeline`] takes care of storing all components (models, schedulers, processors) for diffusion pipelines
    and handles methods for loading, downloading and saving models as well as a few methods common to all pipelines to:

        - move all Paddle modules to the device of your choice
        - enabling/disabling the progress bar for the denoising iteration

    Class attributes:

        - **config_name** ([`str`]) -- name of the config file that will store the class and module names of all
          components of the diffusion pipeline.
    """
    config_name = "model_index.json"

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        from . import pipelines

        for name, module in kwargs.items():
            # retrieve library
            if module is None:
                register_dict = {name: (None, None)}
            else:
                # TODO (junnyu) support paddlenlp.transformers
                if "paddlenlp" in module.__module__.split("."):
                    library = "paddlenlp.transformers"
                else:
                    library = module.__module__.split(".")[0]

                # check if the module is a pipeline module
                pipeline_dir = module.__module__.split(".")[-2] if len(
                    module.__module__.split(".")) > 2 else None
                path = module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(
                    pipelines, pipeline_dir)

                # if library is not in LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                if library not in LOADABLE_CLASSES or is_pipeline_module:
                    library = pipeline_dir

                # retrieve class_name
                class_name = module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        self.save_config(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        # TODO (junnyu) support old version
        model_index_dict.pop("_diffusers_paddle_version", None)
        model_index_dict.pop("_ppdiffusers_version", None)
        model_index_dict.pop("_module", None)

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            if sub_model is None:
                # edge case for saving a pipeline with safety_checker=None
                continue

            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(
                            model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            save_method = getattr(sub_model, save_method_name)
            save_method(os.path.join(save_directory, pipeline_component_name))

    def to(self, paddle_device: Optional[str] = None):
        if paddle_device is None:
            return self

        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, nn.Layer):
                if module.dtype == paddle.float16 and str(paddle_device) in [
                        "cpu"
                ]:
                    logger.warning(
                        "Pipelines loaded with `paddle_dtype=paddle.float16` cannot run with `cpu` device. It"
                        " is not recommended to move them to `cpu` as running them will fail. Please make"
                        " sure to use an accelerator to run the pipeline in inference, due to the lack of"
                        " support for`float16` operations on this device in Paddle. Please, remove the"
                        " `paddle_dtype=paddle.float16` argument, or use another device for inference."
                    )
                module.to(paddle_device)
        return self

    @property
    def device(self):
        r"""
        Returns:
            `paddle.device`: The paddle device on which the pipeline is located.
        """
        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, nn.Layer):
                return module.place
        return "cpu"

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Optional[Union[str,
                                                               os.PathLike]],
            **kwargs):
        r"""
        Instantiate a Paddle diffusion pipeline from pre-trained pipeline weights.

        The pipeline is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained pipeline hosted inside in `https://bj.bcebos.com/paddlenlp/models/community`.
                      like `CompVis/stable-diffusion-v1-4`, `CompVis/ldm-text2im-large-256`.
                    - A path to a *directory* containing pipeline weights saved using
                      [`~DiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load - and saveable variables - *i.e.* the pipeline components - of the
                specific pipeline class. The overwritten components are then directly passed to the pipelines
                `__init__` method. See example below for more information.

        Examples:

        ```py
        >>> from ppdiffusers import DiffusionPipeline

        >>> # Download pipeline from bos and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

        >>> # Use a different scheduler
        >>> from ppdiffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.scheduler = scheduler
        ```
        """
        cache_dir = kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        # (TODO junnyu, we donot suuport this.)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)

        # 1. Download the checkpoints and configs
        if not os.path.isdir(pretrained_model_name_or_path):
            config_dict = cls.load_config(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
            )
        else:
            config_dict = cls.load_config(pretrained_model_name_or_path)

        # 2. Load the pipeline class
        if cls != DiffusionPipeline:
            pipeline_class = cls
        else:
            diffusers_module = importlib.import_module(
                cls.__module__.split(".")[0])
            pipeline_class = getattr(diffusers_module,
                                     config_dict["_class_name"])

        # To be removed in 1.0.0
        # TODO (junnyu) support old version
        _ppdiffusers_version = config_dict[
            "_diffusers_paddle_version"] if "_diffusers_paddle_version" in config_dict else config_dict[
                "_ppdiffusers_version"]
        if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
                version.parse(_ppdiffusers_version).base_version
        ) <= version.parse("0.5.1"):
            from . import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy

            pipeline_class = StableDiffusionInpaintPipelineLegacy

            deprecation_message = (
                "You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the"
                f" {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For"
                " better inpainting results, we strongly suggest using Stable Diffusion's official inpainting"
                " checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting instead or adapting your"
                f" checkpoint {pretrained_model_name_or_path} to the format of"
                " https://huggingface.co/runwayml/stable-diffusion-inpainting. Note that we do not actively maintain"
                " the {StableDiffusionInpaintPipelineLegacy} class and will likely remove it in version 1.0.0."
            )
            deprecate("StableDiffusionInpaintPipelineLegacy",
                      "1.0.0",
                      deprecation_message,
                      standard_warn=False)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules = set(
            inspect.signature(pipeline_class.__init__).parameters.keys()) - set(
                ["self"])
        passed_class_obj = {
            k: kwargs.pop(k)
            for k in expected_modules if k in kwargs
        }

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(
            config_dict, **kwargs)

        if len(unused_kwargs) > 0:
            logger.warning(f"Keyword arguments {unused_kwargs} not recognized.")

        init_kwargs = {}

        # import it here to avoid circular import
        from . import pipelines, ModelMixin

        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            # TODO (junnyu) support old model_index.json
            if library_name == "diffusers_paddle":
                library_name = "ppdiffusers"
            if class_name is None:
                # edge case for when the pipeline was saved with safety_checker=None
                init_kwargs[name] = None
                continue
            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None
            sub_model_should_be_defined = True

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                if not is_pipeline_module and passed_class_obj[name] is not None:
                    library = importlib.import_module(library_name)
                    class_obj = getattr(library, class_name)
                    importable_classes = LOADABLE_CLASSES[library_name]
                    class_candidates = {
                        c: getattr(library, c, None)
                        for c in importable_classes.keys()
                    }

                    expected_class_obj = None
                    for class_name, class_candidate in class_candidates.items():
                        if class_candidate is not None and issubclass(
                                class_obj, class_candidate):
                            expected_class_obj = class_candidate

                    if not issubclass(passed_class_obj[name].__class__,
                                      expected_class_obj):
                        raise ValueError(
                            f"{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be"
                            f" {expected_class_obj}")
                elif passed_class_obj[name] is None:
                    logger.warn(
                        f"You have passed `None` for {name} to disable its functionality in {pipeline_class}. Note"
                        f" that this might lead to problems when using {pipeline_class} and is not recommended."
                    )
                    sub_model_should_be_defined = False
                else:
                    logger.warn(
                        f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                        " has the correct type")

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = getattr(pipeline_module, class_name)
                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = {
                    c: class_obj
                    for c in importable_classes.keys()
                }
            else:
                # else we just import it from the library.
                library = importlib.import_module(library_name)

                class_obj = getattr(library, class_name)
                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {
                    c: getattr(library, c, None)
                    for c in importable_classes.keys()
                }

            if loaded_sub_model is None and sub_model_should_be_defined:
                load_method_name = None
                for class_name, class_candidate in class_candidates.items():
                    if class_candidate is not None and issubclass(
                            class_obj, class_candidate):
                        load_method_name = importable_classes[class_name][1]

                if load_method_name is None:
                    none_module = class_obj.__module__
                    is_dummy_path = none_module.startswith(
                        DUMMY_MODULES_FOLDER) or none_module.startswith(
                            PADDLENLP_DUMMY_MODULES_FOLDER)
                    if is_dummy_path and "dummy" in none_module:
                        # call class_obj for nice error message of missing requirements
                        class_obj()

                    raise ValueError(
                        f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
                        f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
                    )

                load_method = getattr(class_obj, load_method_name)
                loading_kwargs = {}

                if issubclass(class_obj, OnnxRuntimeModel):
                    loading_kwargs["provider"] = provider
                    loading_kwargs["sess_options"] = sess_options

                if issubclass(class_obj, ModelMixin):
                    loading_kwargs["cache_dir"] = cache_dir

                model_path_dir = os.path.join(
                    pretrained_model_name_or_path, name) if os.path.isdir(
                        pretrained_model_name_or_path
                    ) else pretrained_model_name_or_path + "/" + name

                loaded_sub_model = load_method(model_path_dir, **loading_kwargs)

            # TODO junnyu find a better way to covert to float16
            if isinstance(loaded_sub_model, nn.Layer):
                if next(loaded_sub_model.named_parameters()
                        )[1].dtype != paddle_dtype:
                    loaded_sub_model = loaded_sub_model.to(dtype=paddle_dtype)
                # paddlenlp model is training mode not eval mode
                loaded_sub_model.eval()

            init_kwargs[
                name] = loaded_sub_model  # UNet(...), # DiffusionScheduler(...)

        # 4. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        if len(missing_modules) > 0 and missing_modules <= set(
                passed_class_obj.keys()):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj[module]
        elif len(missing_modules) > 0:
            passed_modules = set(
                list(init_kwargs.keys()) + list(passed_class_obj.keys()))
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 5. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)
        return model

    @property
    def components(self) -> Dict[str, Any]:
        r"""

        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations to not have to re-allocate memory.

        Examples:

        ```py
        >>> from ppdiffusers import (
        ...     StableDiffusionPipeline,
        ...     StableDiffusionImg2ImgPipeline,
        ...     StableDiffusionInpaintPipeline,
        ... )

        >>> text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        ```

        Returns:
            A dictionaly containing all the modules needed to initialize the pipeline.
        """
        components = {
            k: getattr(self, k)
            for k in self.config.keys() if not k.startswith("_")
        }
        expected_modules = set(
            inspect.signature(self.__init__).parameters.keys()) - set(["self"])

        if set(components.keys()) != expected_modules:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        return tqdm(iterable, **self._progress_bar_config)

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs
