# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import textwrap
import inspect

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

from paddlenlp.utils.env import PPNLP_HOME


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


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

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp)


def _get_source_code(function, dedent=True):
    source_code = inspect.getsource(function)
    if dedent:
        source_code = textwrap.dedent(source_code)
    return source_code


def _write_setup_file(name, sourcedir, file_path):
    """
    Automatically generate setup.py and write it into build directory.
    """
    setup = textwrap.dedent("""
    setup(
        name='{name}',
        ext_modules=[
            CMakeExtension(
                name={name},
                sourcedir={sourcedir})],
        cmdclass={{"build_ext" : CMakeBuild
        }})""").lstrip().format(
        name=name,
        sourcedir=sourcedir,
        cmake_ext=_get_source_code(CMakeExtension),
        cmake_build=_get_source_code(CMakeBuild))

    template = textwrap.dedent("""
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext

    {cmake_ext}

    {cmake_build}

    setup(
        name='{name}',
        ext_modules=[
            CMakeExtension(
                name={name},
                sourcedir={sourcedir})],
        cmdclass={{"build_ext" : CMakeBuild
        }})""").lstrip()

    content = template.format(
        name=name,
        sourcedir=sourcedir,
        cmake_ext=_get_source_code(CMakeExtension),
        cmake_build=_get_source_code(CMakeBuild))

    # print('write setup.py into {}'.format(file_path), verbose)
    with open(file_path, 'w') as f:
        f.write(content)


def load():
    # write setup file
    #_write_setup_file()
    # jit compile
    #_jit_compile()
    pass


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="cmake_example",
    version="0.0.1",
    author="guosheng",
    author_email="guosheng@baidu.com",
    description="A test project for setup using CMake",
    long_description="",
    ext_modules=[CMakeExtension(
        "cmake_example", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False, )
