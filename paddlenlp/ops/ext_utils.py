# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from filelock import FileLock
from paddle.utils.cpp_extension import load_op_meta_info_and_register_op
from paddle.utils.cpp_extension.cpp_extension import CUDA_HOME
from paddle.utils.cpp_extension.cpp_extension import (
    BuildExtension as PaddleBuildExtension,
)
from paddle.utils.cpp_extension.cpp_extension import CppExtension
from paddle.utils.cpp_extension.extension_utils import (
    _import_module_from_library,
    _jit_compile,
)
from setuptools import Extension

from paddlenlp.utils.env import PPNLP_HOME
from paddlenlp.utils.log import logger

if CUDA_HOME and not os.path.exists(CUDA_HOME):
    # CUDA_HOME is only None for Windows CPU version in paddle `find_cuda_home`.
    # Clear it for other non-CUDA situations.
    CUDA_HOME = None

LOADED_EXT = {}


def file_lock(lock_file_path):
    def _wrapper(func):
        @functools.wraps(func)
        def _impl(*args, **kwargs):
            with FileLock(lock_file_path):
                func(*args, **kwargs)

        return _impl

    return _wrapper


def _get_files(path):
    """
    Helps to list all files under the given path.
    """
    if os.path.isfile(path):
        return [path]
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        for file in files:
            file = os.path.join(root, file)
            all_files.append(file)
    return all_files


# copy form distutils.dep_util to avoid import distutils
def newer_group(sources, target, missing="error"):
    """Return true if 'target' is out-of-date with respect to any file
    listed in 'sources'.  In other words, if 'target' exists and is newer
    than every file in 'sources', return false; otherwise return true.
    'missing' controls what we do when a source file is missing; the
    default ("error") is to blow up with an OSError from inside 'stat()';
    if it is "ignore", we silently drop any missing source files; if it is
    "newer", any missing source files make us assume that 'target' is
    out-of-date (this is handy in "dry-run" mode: it'll make you pretend to
    carry out commands that wouldn't work because inputs are missing, but
    that doesn't matter because you're not actually going to run the
    commands).
    """
    # If the target doesn't even exist, then it's definitely out-of-date.
    if not os.path.exists(target):
        return 1

    # Otherwise we have to find out the hard way: if *any* source file
    # is more recent than 'target', then 'target' is out-of-date and
    # we can immediately return true.  If we fall through to the end
    # of the loop, then 'target' is up-to-date and we return false.
    from stat import ST_MTIME

    target_mtime = os.stat(target)[ST_MTIME]
    for source in sources:
        if not os.path.exists(source):
            if missing == "error":  # blow up when we stat() the file
                pass
            elif missing == "ignore":  # missing source dropped from
                continue  # target's dependency list
            elif missing == "newer":  # missing source means target is
                return 1  # out-of-date

        source_mtime = os.stat(source)[ST_MTIME]
        if source_mtime > target_mtime:
            return 1
    else:
        return 0


class CMakeExtension(Extension):
    def __init__(self, name, source_dir=None):
        # A CMakeExtension needs a source_dir instead of a file list.
        Extension.__init__(self, name, sources=[])
        if source_dir is None:
            self.source_dir = str(Path(__file__).parent.resolve())
        else:
            self.source_dir = os.path.abspath(os.path.expanduser(source_dir))
        self.sources = _get_files(self.source_dir)

    def build_with_command(self, ext_builder):
        """
        Custom `build_ext.build_extension` in `Extension` instead of `Command`.
        `ext_builder` is the instance of `build_ext` command.
        """
        # refer to https://github.com/pybind/cmake_example/blob/master/setup.py
        if ext_builder.compiler.compiler_type == "msvc":
            raise NotImplementedError
        cmake_args = getattr(self, "cmake_args", []) + [
            "-DCMAKE_BUILD_TYPE={}".format("Debug" if ext_builder.debug else "Release"),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_builder.build_lib),
        ]
        build_args = []

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(ext_builder, "parallel") and ext_builder.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(ext_builder.parallel)]

        if not os.path.exists(ext_builder.build_temp):
            os.makedirs(ext_builder.build_temp)

        # Redirect stdout/stderr to mute, especially when allowing errors
        stdout = getattr(self, "_std_out_handle", None)
        subprocess.check_call(
            ["cmake", self.source_dir] + cmake_args, cwd=ext_builder.build_temp, stdout=stdout, stderr=stdout
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=ext_builder.build_temp, stdout=stdout, stderr=stdout
        )

    def get_target_filename(self):
        """
        The file names of libraries. Currently only support one library for
        one extension.
        """
        raise NotImplementedError

    def get_output_filename(self):
        """
        The file names of outputs, which mostly is the same with
        `get_target_filename`.
        """
        return self.get_target_filename()


class FasterTransformerExtension(CMakeExtension):
    def __init__(self, name, source_dir=None, need_parallel=False):
        super(FasterTransformerExtension, self).__init__(name, source_dir)
        self.sources = _get_files(os.path.join(self.source_dir, "fast_transformer", "src")) + _get_files(
            os.path.join(self.source_dir, "patches", "FasterTransformer")
        )
        self._std_out_handle = None
        # Env variable may not work as expected, since jit compile by `load`
        # would not re-built if source code is not update.
        # self.sm = os.environ.get("PPNLP_GENERATE_CODE", None)
        # Whether or not to use model parallel. Note that since the building use
        # a new process, we shoud find a way to let it know whether to use model
        # parallel.
        self.need_parallel = need_parallel

    def build_with_command(self, ext_builder):
        if CUDA_HOME is None:  # GPU only
            # TODO(guosheng): should we touch a dummy file or add a quick exit
            # method to avoid meaningless process in `load`
            logger.warning("FastGeneration is not available because CUDA can not be found.")
            raise NotImplementedError
        # TODO(guosheng): Multiple -std seems be passed in FastGeneration,
        # which is not allowed by NVCC. Fix it later.
        self.cmake_args = [f"-DPY_CMD={sys.executable}"]
        # `GetCUDAComputeCapability` is not exposed yet, and detect CUDA/GPU
        # version in cmake file.
        # self.cmake_args += [f"-DSM={self.sm}"] if self.sm is not None else []
        self.cmake_args += ["-DWITH_GPT=ON"]
        if self.need_parallel:
            self.cmake_args += ["-DWITH_PARALLEL=ON"]
        try:
            super(FasterTransformerExtension, self).build_with_command(ext_builder)
            # FastGeneration cmake file resets `CMAKE_LIBRARY_OUTPUT_DIRECTORY`
            # to `CMAKE_BINARY_DIR/lib`, thus copy the lib back to `build_ext.build_lib`.
            # Maybe move this copy to CMakeList.
            # `copy_tree` or `copy_file`, boost lib might be included
            ext_builder.copy_tree(os.path.join(ext_builder.build_temp, "lib"), ext_builder.build_lib)
            # TODO(guosheng): Maybe we should delete the build dir especially
            # when it is in the dir of paddlenlp package.
            # os.remove(ext_builder.build_temp)
        except Exception as e:
            logger.warning("FastGeneration is not available due to build errors.")
            raise e

    def get_target_filename(self):
        # CMake file has fixed the name of lib, maybe we can copy it as the name
        # returned by `BuildExtension.get_ext_filename` after build.
        return "libdecoding_op.so"

    def get_output_filename(self):
        return "libdecoding_op.so"


class BuildExtension(PaddleBuildExtension):
    """
    Support both `CppExtention` of Paddle and custom extensions of PaddleNLP.
    """

    def build_extensions(self):
        custom_exts = []  # for
        no_custom_exts = []  # for normal extentions paddle.utils.cpp_extension
        for ext in self.extensions:
            if hasattr(ext, "build_with_command"):
                # custom build in Extension
                ext.build_with_command(self)
                custom_exts.append(ext)
            else:
                no_custom_exts.append(ext)
        if no_custom_exts:
            # Build CppExtentio/CUDAExtension with `PaddleBuildExtension`
            self.extensions = no_custom_exts
            super(BuildExtension, self).build_extensions()
        self.extensions = custom_exts + no_custom_exts


EXTENSIONS = {
    "FastGeneration": FasterTransformerExtension,
    # NOTE: Since model parallel code is supported by definitions, to avoid
    # performance degrading on non-parallel mode, we use a separated lib for
    # model parallel.
    "FasterTransformerParallel": FasterTransformerExtension,
}


def get_extension_maker(name):
    # Use `paddle.utils.cpp_extension.CppExtension` as the default
    # TODO(guosheng): Maybe register extension classes into `Extensions`.
    return EXTENSIONS.get(name, CppExtension)


def _write_setup_file(name, file_path, build_dir, **kwargs):
    """
    Automatically generate setup.py and write it into build directory.
    `kwargws` is arguments for the corresponding Extension initialization.
    Any type extension can be jit build.
    """
    template = textwrap.dedent(
        """
    from setuptools import setup
    from paddlenlp.ops.ext_utils import get_extension_maker, BuildExtension

    setup(
        name='{name}',
        ext_modules=[
            get_extension_maker('{name}')(
                name='{name}',
                {kwargs_str})],
        cmdclass={{'build_ext' : BuildExtension.with_options(
            output_dir=r'{build_dir}')
        }})"""
    ).lstrip()
    kwargs_str = ""
    for key, value in kwargs.items():
        kwargs_str += key + "=" + (f"'{value}'" if isinstance(value, str) else str(value)) + ","
    content = template.format(name=name, kwargs_str=kwargs_str, build_dir=build_dir)

    with open(file_path, "w") as f:
        f.write(content)


@file_lock(os.path.join(PPNLP_HOME, "load_ext.lock"))
def load(name, build_dir=None, force=False, verbose=False, **kwargs):
    # TODO(guosheng): Need better way to resolve unsupported such as CPU. Currently,
    # raise NotImplementedError and skip `_jit_compile`. Otherwise, `_jit_compile`
    # will output the error to stdout (when verbose is True) and raise `RuntimeError`,
    # which is not friendly for users though no other bad effect.
    if CUDA_HOME is None:
        logger.warning("%s is not available because CUDA can not be found." % name)
        raise NotImplementedError
    if name in LOADED_EXT.keys():
        # TODO(guosheng): Maybe the key should combined with kwargs since the
        # extension object is created using them.
        return LOADED_EXT[name]
    if build_dir is None:
        # build_dir = os.path.join(PPNLP_HOME, 'extenstions')
        # Maybe under package dir is better to avoid cmake source path conflict
        # with different source path, like this:
        # build_dir = os.path.join(
        #     str(Path(__file__).parent.resolve()), 'extenstions')
        # However if it is under the package dir, it might make the package hard
        # to uninstall. Thus we put it in PPNLP_HOME with digest of current path,
        # like this:
        build_dir = os.path.join(
            PPNLP_HOME, "extensions", hashlib.md5(str(Path(__file__).parent.resolve()).encode("utf-8")).hexdigest()
        )
    build_base_dir = os.path.abspath(os.path.expanduser(os.path.join(build_dir, name)))
    if not os.path.exists(build_base_dir):
        os.makedirs(build_base_dir)

    extension = get_extension_maker(name)(name, **kwargs)
    # Check if 'target' is out-of-date with respect to any file to avoid rebuild
    if isinstance(extension, CMakeExtension):
        # `CppExtention/CUDAExtension `has version manager by `PaddleBuildExtension`
        # Maybe move this to CMakeExtension later.
        # TODO(guosheng): flags/args changes may also trigger build, and maybe
        # need version manager like `PaddleBuildExtension`.
        out_filename = extension.get_output_filename()
        if isinstance(out_filename, str):
            out_filename = [out_filename]
        out_filepath = [os.path.join(build_base_dir, f) for f in out_filename]
        lib_filename = extension.get_target_filename()
        lib_filepath = os.path.join(build_base_dir, lib_filename)
        if not force:
            ext_sources = extension.sources
            if all(os.path.exists(f) and not newer_group(ext_sources, f, "newer") for f in out_filepath):
                logger.debug("skipping '%s' extension (up-to-date) build" % name)
                ops = load_op_meta_info_and_register_op(lib_filepath)
                LOADED_EXT[name] = ops
                return LOADED_EXT[name]

    # write setup file and jit compile
    file_path = os.path.join(build_dir, name, "{}_setup.py".format(name))
    _write_setup_file(name, file_path, build_base_dir, **kwargs)
    _jit_compile(file_path, verbose)
    if isinstance(extension, CMakeExtension):
        # Load a shared library (if exists) only to register op.
        if os.path.exists(lib_filepath):
            ops = load_op_meta_info_and_register_op(lib_filepath)
            LOADED_EXT[name] = ops
            return LOADED_EXT[name]
    else:
        # Import as callable python api
        return _import_module_from_library(name, build_base_dir, verbose)
