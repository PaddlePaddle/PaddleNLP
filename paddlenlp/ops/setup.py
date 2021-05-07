# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import textwrap
import inspect

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.dep_util import newer_group

from paddle.utils.cpp_extension.extension_utils import _jit_compile
from paddle.utils.cpp_extension.cpp_extension import CUDA_HOME
from paddlenlp.utils.env import PPNLP_HOME
from paddlenlp.utils.log import logger

if not os.path.exists(CUDA_HOME):
    # CUDA_HOME is only None when `core.is_compiled_with_cuda()` is True in
    # find_cuda_home. Clear it for paddle cpu version.
    CUDA_HOME = None


# A CMakeExtension needs a source_dir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, source_dir=""):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            # "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            # "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DPY_CMD={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(
                cfg),  # not used on MSVC, but no harm
        ]
        build_args = []

        if self.compiler.compiler_type == "msvc":
            raise NotImplementedError

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # TODO(guosheng): Multiple -std seems be passed in FasterTransformer,
        # which is not allowed by NVCC. Fix it later.
        # TODO(guosheng): Redirect stdout/stderr and handle errors
        subprocess.check_call(
            ["cmake", ext.source_dir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp)
        # copy to


def _get_source_code(function, dedent=True):
    source_code = inspect.getsource(function)
    if dedent:
        source_code = textwrap.dedent(source_code)
    return source_code


def _build_and_run():
    # CMAKE or paddle GetCUDAComputeCapability
    pass


def _write_setup_file(name, source_dir, file_path, build_dir):
    """
    Automatically generate setup.py and write it into build directory.
    """
    template = textwrap.dedent("""
    from setuptools import setup
    from paddlenlp.ops.ext_utils import CMakeExtension, CMakeBuild

    setup(
        name='{name}',
        ext_modules=[
            CMakeExtension(
                name='{name}',
                source_dir='{source_dir}')],
        cmdclass={{"build_ext" : CMakeBuild
        }})""").lstrip()
    content = template.format(name=name, source_dir=source_dir)

    with open(file_path, 'w') as f:
        f.write(content)


def _write_error(e, error_file):
    data = {
        "message": {
            "message": f"{type(e).__name__}: {e}",
            "extraInfo": {
                "py_callstack": traceback.format_exc(),
                "timestamp": str(int(time.time())),
            },
        }
    }

    if error_file:
        with open(error_file, "w") as fp:
            json.dump(data, fp)
    else:
        log.error(json.dumps(data, indent=2))


def run_cmd(command, verbose=False):
    """
    Execute command with subprocess.
    """
    # logging
    log_v("execute command: {}".format(command), verbose)
    try:
        from subprocess import DEVNULL  # py3
    except ImportError:
        DEVNULL = open(os.devnull, 'wb')

    # execute command
    try:
        if verbose:
            return subprocess.check_call(
                command, shell=True, stderr=subprocess.STDOUT)
        else:
            return subprocess.check_call(command, shell=True, stdout=DEVNULL)
    except Exception:
        _, error, _ = sys.exc_info()
        raise RuntimeError("Failed to run command: {}, errors: {}".format(
            compile, error))


def load(name, source_dir=None, build_dir=None, force=False, verbose=False):
    if source_dir is None:
        source_dir = os.path.dirname(__file__)
    if build_dir is None:
        build_dir = os.path.join(PPNLP_HOME, 'extenstions')
    build_base_dir = os.path.join(build_dir, name)
    source_dir = os.path.expanduser(source_dir)
    build_base_dir = os.path.expanduser(build_base_dir)
    if not os.path.exists(build_base_dir):
        os.makedirs(build_base_dir)
        force = True
    elif not os.listdir(build_base_dir):
        force = True
    else:
        # TODO(guosheng): use name argument as the name of lib
        ext_name = os.listdir(build_base_dir)[0]
        ext_path = os.path.join(build_base_dir, ext_name)
    # Check if 'target' is out-of-date with respect to any file to avoid rebuild
    sources = os.listdir(source_dir)
    if not (force or newer_group(sources, ext_path, 'newer')):
        logger.debug("skipping '%s' extension (up-to-date)", name)
        return
    # write setup file
    file_path = os.path.join(build_dir, "{}_setup.py".format(name))
    _write_setup_file(name, source_dir, file_path, build_base_dir)
    # jit compile
    try:
        _jit_compile(file_path, verbose)
    except Exception as e:
        logger.warn("Failed to compile extension %s, would .")
        # record exception for debug
        _, error, _ = sys.exc_info()
        _write_error(e, error_file)
        raise RuntimeError("Failed to run command: {}, errors: {}".format(
            compile, error))


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
# setup(
#     name="cmake_example",
#     version="0.0.1",
#     author="paddlesl",
#     author_email="paddlesl@baidu.com",
#     description="A test project for setup using CMake",
#     long_description="",
#     ext_modules=[CMakeExtension("cmake_example", source_dir=".")],
#     cmdclass={"build_ext": CMakeBuild},
#     package_data={},
#     zip_safe=False,
# )
