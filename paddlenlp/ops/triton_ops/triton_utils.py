# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import os
import re
import sys

import paddle
import triton
from paddle.base.framework import OpProtoHolder

compile_file = triton.__path__[0] + "/tools/compile.py"
link_file = triton.__path__[0] + "/tools/link.py"
python_path = sys.executable


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


def find_so_path(generated_dir, python_package_name):
    """
    find the specified so in generated_dir, if not found it will return None.
    """

    so_path = []
    for root, dirs, files in os.walk(generated_dir):
        for file in files:
            if file.endswith(python_package_name + ".so"):
                so_path.append(os.path.join(root, file))
    if len(so_path) == 0:
        return None
    else:
        assert len(so_path) == 1
        return so_path[0]


def multi_process_do(commands):
    THREADS = 40
    import multiprocessing

    process = []

    def one_process_work(commands, thread_id):
        i = thread_id
        while i < len(commands):
            re = os.system(commands[i])
            assert re == 0
            i += THREADS

    for i in range(THREADS):
        p = multiprocessing.Process(target=one_process_work, args=(commands, i))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()


def extract_triton_kernel(kernel, file_name):
    """
    Extract the triton kernel and write it to the specified file_name.

    Args:
        kernel: the triton kernel name.
        file_name: the file name you want to write.
    """

    import inspect
    import re
    import textwrap

    fn = kernel
    if type(kernel) == triton.runtime.jit.JITFunction:
        fn = kernel.fn
    elif type(kernel) == triton.runtime.autotuner.Autotuner:
        fn = kernel.fn.fn
    else:
        AssertionError("error occures")
    py_script = textwrap.dedent(inspect.getsource(fn))

    # @triton.jit must only appear once
    # assert len(re.findall("@triton.jit", py_script)) == 1
    assert len(re.findall("def ", py_script)) == 1
    # assert len(re.findall("@haha()", py_script)) == 1
    # py_script = py_script.replace("@haha()", "@triton.jit")

    py_script = py_script[py_script.find("def ") :]
    py_script = "import triton\nimport triton.language as tl\n\n\n@triton.jit\n" + py_script

    py_script = py_script.replace("if bias_ptr is not None", "if bias_ptr")

    with open(file_name, "w") as f:
        f.write(py_script)
        f.close()


template_install = """

import os
generated_cu = []
for root, dirs, files in os.walk("./"):
    for file in files:
        if file.endswith(".c") or file.endswith(".cu"):
            generated_cu.append(os.path.join(root, file))


import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{{0}},code=sm_{{0}}".format(cc)]


gencode_flags = get_gencode_flags()



setup(
    name="{python_package_name}",
    ext_modules=CUDAExtension(
        sources = generated_cu,
        extra_compile_args={{
            "cc": ["-lcuda"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ]
            + gencode_flags,
        }},
        extra_link_args = ["-lcuda"]
    ),
)
"""


def get_op_name_with_suffix(op_name, x_list):
    suffix = []
    for x in x_list:
        if x % 16 == 0:
            suffix.append(16)
        elif x == 1:
            suffix.append(1)
        else:
            suffix.append(0)
    return op_name + "_".join([str(i) for i in suffix])


def get_value_hint(x):
    hint = ""
    for ele in x:
        if type(ele) == int:
            if ele % 16 == 0 and ele > 0:
                hint += "i64:16,"
            elif ele == 1:
                hint += "i64:1,"
            else:
                hint += "i64,"
        if type(ele) == float:
            hint += "fp32,"
    return hint


def get_dtype_str(dtype):
    if dtype == paddle.float16:
        return "_fp16"
    if dtype == paddle.float8_e4m3fn:
        return "_float8_e4m3fn"
    elif dtype == paddle.uint8:
        return "_u8"
    elif dtype == paddle.int8:
        return "_i8"
    elif dtype == paddle.int32:
        return "_i32"
    elif dtype == paddle.int64:
        return "_i64"
    elif dtype == paddle.float32:
        return "_fp32"
    elif dtype == paddle.bfloat16:
        return "_bf16"
    else:
        raise ValueError("Not support this dtype.")


def build_package(generated_dir, python_package_name):
    """
    Build the package, not install it.

    Args:
        generated_dir: the source cu file dir.
        python_package_name: the python package name.
    """
    setup_file_path = generated_dir + "/setup_cuda.py"
    python_path = sys.executable
    with open(setup_file_path, "w") as f:
        f.write(template_install.format(python_package_name=python_package_name))
        f.close()
    install_command = f"cd {generated_dir} && {python_path} setup_cuda.py build"
    re = os.system(install_command)
    assert re == 0


def rename_c_to_cu(generated_dir):
    """
    Rename the .c files int generated_dir to .cu file, because the triton aot tool generate the .c files.
    """
    # rename the .c file to .cu
    for filename in os.listdir(generated_dir):
        if filename.endswith(".c"):
            old_path = os.path.join(generated_dir, filename)
            new_path = os.path.join(generated_dir, filename + "u")
            os.rename(old_path, new_path)


def get_pointer_hint(dtypes):
    hint = ""
    for ele in dtypes:
        if ele == paddle.float16:
            hint += "*fp16:16,"
        elif ele == paddle.uint8:
            hint += "*u8:16,"
        elif ele == paddle.int8:
            hint += "*i8:16,"
        elif ele == paddle.float32:
            hint += "*fp32:16,"
        elif ele == paddle.bfloat16:
            hint += "*bf16:16,"
        elif ele == paddle.int32:
            hint += "*i32:16,"
        elif ele == paddle.int64:
            hint += "*i64,"
        elif ele == paddle.float8_e4m3fn:
            hint += "*fp8e4nv:16,"
    return hint


paddle_custom_op_head_part = """#include <vector>
#include <map>
#include "${op_name}_kernel.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_${op_name};

CUdeviceptr get_tensor_ptr(const paddle::Tensor& input){
  if (input.type() == paddle::DataType::FLOAT16) {
    return (CUdeviceptr)(input.data<phi::dtype::float16>());
  } else if (input.type() == paddle::DataType::BFLOAT16) {
    return (CUdeviceptr)(input.data<phi::dtype::bfloat16>());
  } else if (input.type() == paddle::DataType::INT32) {
    return (CUdeviceptr)(input.data<int>());
  } else if (input.type() == paddle::DataType::FLOAT32) {
    return (CUdeviceptr)(input.data<float>());
  } else if (input.type() == paddle::DataType::UINT8) {
    return (CUdeviceptr)(input.data<uint8_t>());
  } else if (input.type() == paddle::DataType::INT8) {
    return (CUdeviceptr)(input.data<int8_t>());
  } else if (input.type() == paddle::DataType::INT64) {
    return (CUdeviceptr)(input.data<int64_t>());
  } else if (input.type() == paddle::DataType::INT32) {
    return (CUdeviceptr)(input.data<int32_t>());
  } else if (input.type() == paddle::DataType::FLOAT8_E4M3FN) {
    return (CUdeviceptr)(input.data<phi::dtype::float8_e4m3fn>());
  } else {
    assert(false);
    return (CUdeviceptr)(nullptr);
  }
}

int triton_cdiv(int x, int y) {
    int result = (x + y - 1) / y;
    return (int)(result);
}
"""

tune_and_invoke_part = """
  std::vector<int> problem_size = {${key}};
  auto run_triton_kernel = [&](int algo_id) -> CUresult{
      return ${op_name}_kernel(run_stream,
                                               ${triton_kernel_args},
                                               algo_id);
  };

  map_problem_${op_name}[problem_size] = 0;

  if (!map_problem_${op_name}.count(problem_size)) {
    std::cout << "we are tuning for ${op_name} which key is: {";
    for (int i = 0; i < problem_size.size(); i++) {
        std::cout << problem_size[i] << ", ";
    }
    std::cout << "}" << std::endl;

    float min_time = 10000.f;
    int select_id = -1;
    constexpr int WARMUP = 5;
    constexpr int REPEAT = 10;

    for (int algo_id = 0; algo_id < ${op_name}_kernel_get_num_algos(); ++algo_id) {
        cudaEvent_t beg[REPEAT];
        cudaEvent_t end[REPEAT];
        float elapsed_times[REPEAT];

        auto status = CUDA_SUCCESS;

        for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
            int repeat_id = ii - WARMUP;

            if (repeat_id >= 0) {
                (cudaEventCreate(beg + repeat_id));
                (cudaEventCreate(end + repeat_id));
                (cudaEventRecord(beg[repeat_id]));
            }

            auto flush_l2_cache = paddle::full(
                {10 * 1024 * 1024}, 0, paddle::DataType::INT32, ${arbitary_output_name}.place());
            // std::cout << &flush_l2_cache  << std::endl;
            // this is used when out is need to be reset to zero, such as split-k gemm.
            ${reset_zero_when_tune};

            status = run_triton_kernel(algo_id);
            // assert(status == CUDA_SUCCESS);

            if (repeat_id >= 0) {
                (cudaEventRecord(end[repeat_id]));
                (cudaEventSynchronize(end[repeat_id]));
                (cudaEventElapsedTime(
                    elapsed_times + repeat_id, beg[repeat_id], end[repeat_id]));
            }
        }

        float avg_elapsed_time = 0.f;
        for (int ii = 0; ii < REPEAT; ++ii) {
            avg_elapsed_time += elapsed_times[ii];
        }

        std::cout << "algo id " << algo_id << " costs " << avg_elapsed_time << " ms" << std::endl;

        if (avg_elapsed_time < min_time && status == CUDA_SUCCESS) {
            min_time = avg_elapsed_time;
            select_id = algo_id;
        }
    }

    map_problem_${op_name}[problem_size] = select_id;
    std::cout << "select algo id: " << select_id << std::endl;
    ${reset_zero_when_tune};
  }

  if (map_problem_${op_name}.count(problem_size)) {
    int algo_id = map_problem_${op_name}[problem_size];
    auto status = run_triton_kernel(algo_id);
    assert(status == CUDA_SUCCESS);
  }
"""


common_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(${input_and_attr}) {
  ${prepare_attr_for_triton_kernel}
  ${prepare_ptr_for_triton_kernel}
  auto  run_stream = ${arbitary_output_name}.stream();
  """
    + tune_and_invoke_part
    + """
  return {${return_tensor_names}};
}

${d2s_infer_code}

PD_BUILD_OP(${op_name})
    .Inputs({${paddle_input_sig}})
    .Outputs({${paddle_output_sig}})
    .Attrs({${paddle_attr_sig}})
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
"""
)


def rendering_common_template(
    func,
    prepare_attr_for_triton_kernel,
    prepare_ptr_for_triton_kernel,
    return_tensor_names=None,
    d2s_infer_code="",
):
    signature = inspect.signature(func)
    arg_names = [v.name for v in signature.parameters.values()]
    arg_defaults = [v.default for v in signature.parameters.values()]
    input_and_attr = ""
    paddle_input_sig = ""
    paddle_attr_sig = ""

    if return_tensor_names is None:
        return_tensor_names = "useless"
        prepare_ptr_for_triton_kernel += (
            "auto useless = paddle::empty({1}, paddle::DataType::INT32, paddle::CPUPlace());"
        )

    for i in range(len(arg_names)):
        if arg_defaults[i] is None:
            input_and_attr += f"paddle::optional<paddle::Tensor> & {arg_names[i]},"
            paddle_input_sig += f"""paddle::Optional("{arg_names[i]}"),"""
        elif type(arg_defaults[i]) == float:
            input_and_attr += f"float {arg_names[i]},"
            paddle_attr_sig += f""""{arg_names[i]}: float","""
        elif type(arg_defaults[i]) == bool:
            input_and_attr += f"bool {arg_names[i]},"
            paddle_attr_sig += f""""{arg_names[i]}: bool","""
        elif type(arg_defaults[i]) == int:
            input_and_attr += f"int64_t {arg_names[i]},"
            paddle_attr_sig += f""""{arg_names[i]}: int64_t","""
        elif type(arg_defaults[i]) == str:
            input_and_attr += f"std::string {arg_names[i]},"
            paddle_attr_sig += f""""{arg_names[i]}: std::string","""
        elif arg_names[i] == "config":
            continue
        else:
            input_and_attr += f"const paddle::Tensor & {arg_names[i]},"
            paddle_input_sig += f""""{arg_names[i]}","""
    input_and_attr = input_and_attr[:-1]
    paddle_input_sig = paddle_input_sig[:-1]
    if len(paddle_attr_sig) > 1:
        paddle_attr_sig = paddle_attr_sig[:-1]

    paddle_output_sig = ""
    arbitary_output_name = ""
    for name in return_tensor_names.split(","):
        name = name.strip()
        arbitary_output_name = name
        paddle_output_sig += f""""{name}","""
    paddle_output_sig = paddle_output_sig[:-1]

    if "${op_name}_InferShape" not in d2s_infer_code:
        d2s_infer_shape_part = "std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& A_shape) {return {${tmp}};}\n "
        tmp = ",".join(["A_shape"] * len(return_tensor_names.split(",")))
        tmp_dict = {"tmp": tmp}
        d2s_infer_shape_part = SubstituteTemplate(d2s_infer_shape_part, tmp_dict)

        d2s_infer_code += d2s_infer_shape_part

    if "${op_name}_InferDtype" not in d2s_infer_code:
        d2s_infer_dtype_part = "std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {return {${tmp}};}\n "
        tmp = ",".join(["A_dtype"] * len(return_tensor_names.split(",")))
        tmp_dict = {"tmp": tmp}
        d2s_infer_dtype_part = SubstituteTemplate(d2s_infer_dtype_part, tmp_dict)

        d2s_infer_code += d2s_infer_dtype_part

    result_str = SubstituteTemplate(
        common_template,
        {
            "input_and_attr": input_and_attr,
            "prepare_attr_for_triton_kernel": prepare_attr_for_triton_kernel,
            "prepare_ptr_for_triton_kernel": prepare_ptr_for_triton_kernel,
            "return_tensor_names": return_tensor_names,
            "arbitary_output_name": arbitary_output_name,
            "d2s_infer_code": d2s_infer_code,
            "paddle_input_sig": paddle_input_sig,
            "paddle_output_sig": paddle_output_sig,
            "paddle_attr_sig": paddle_attr_sig,
        },
    )

    return paddle_custom_op_head_part + result_str


class KernelInterface:
    def __init__(
        self,
        func,
        other_config,
        key_args=["1"],
    ):
        self.func = func
        self.key_args = key_args

        signature = inspect.signature(func)
        self.arg_names = [v.name for v in signature.parameters.values()]
        for ele in self.arg_names:
            assert self.arg_names.count(ele) == 1
        # arg_defaults = [v.default for v in signature.parameters.values()]

        # self.annotations = {
        #     name: ty for name, ty in func.__annotations__.items()
        # }
        self.annotations = dict(func.__annotations__)

        self.constexprs = [
            self.arg_names.index(name)
            for name in self.arg_names
            if self.annotations.get(name) == triton.language.core.constexpr
        ]

        self.arg_exclude_constexpr = [
            self.arg_names[i] for i in range(len(self.arg_names)) if i not in self.constexprs
        ]

        import textwrap

        py_script = textwrap.dedent(inspect.getsource(func))

        import re

        pat = r"def\s" + func.__name__
        func_begin = re.findall(pat, py_script)
        assert len(func_begin) == 1
        func_begin = func_begin[0]
        py_script = py_script[py_script.find(func_begin) :]

        def decorator(*args, **kwargs):
            all_input = []

            for i in range(len(args)):
                all_input.append(args[i])

            position_arguments_num = len(all_input)
            for i in range(position_arguments_num, len(self.arg_names)):
                if self.arg_names[i] in kwargs.keys():
                    all_input.append(kwargs[self.arg_names[i]])
                else:
                    # means this input is not specified, it muse be a tl.constexpr.
                    assert i in self.constexprs
                    all_input.append(None)

            dtypes = []
            x_list = []
            const_args = [self.arg_names[i] for i in self.constexprs]
            # we dont allow there are two strings in const_args, and one is a substring of the other.
            for i in const_args:
                for j in const_args:
                    if i != j and i.find(j) != -1:
                        raise ValueError(
                            f"We find {i}, {j} in tl.constexpr args, and {j} is a substring of {i}, please modify your triton kernel arguments names to avoid this."
                        )

            modified_arg_exclude_constexpr = self.arg_exclude_constexpr
            const_hint_dict = {}
            for i in range(len(all_input)):
                ele = all_input[i]
                if (
                    type(ele) == paddle.Tensor
                    or type(ele) == paddle.base.framework.EagerParamBase
                    or type(ele) == paddle.base.framework.Parameter
                    or type(ele) == paddle.base.framework.Variable
                    or type(ele) == paddle.base.libpaddle.pir.Value
                ):
                    dtypes.append(ele.dtype)
                    modified_arg_exclude_constexpr[i] = f"input_ptrs[{i}]"
                elif i in self.constexprs:
                    const_hint_dict[self.arg_names[i]] = ele
                else:
                    x_list.append(ele)

            op_name = self.op_name

            python_package_name = f"{op_name}_package"

            generated_dir = os.getenv("TRITON_KERNEL_CACHE_DIR", None)
            print("the kernel cache dir is:", generated_dir)
            assert (
                generated_dir is not None
            ), "TRITON_KERNEL_CACHE_DIR is None, please set it such as export TRITON_KERNEL_CACHE_DIR=/tmp/haha "
            generated_dir = f"{generated_dir}/{op_name}"
            os.makedirs(generated_dir, exist_ok=True)

            py_script_file = f"{generated_dir}/triton_kernels.py"
            extract_triton_kernel(func, py_script_file)

            address_hint = get_pointer_hint(dtypes)
            value_hint = get_value_hint(x_list)
            const_args = [f"{{{ele}}}" for ele in const_args]
            const_args = ",".join(const_args)

            lanuch_grid = list(self.grid)
            for i in range(len(lanuch_grid)):
                ele = lanuch_grid[i]
                if type(ele) == str:
                    for key in const_hint_dict.keys():
                        if key in ele:
                            ele = ele.replace(key, f"{{{key}}}")
                else:
                    ele = str(ele)

                lanuch_grid[i] = ele
            if len(lanuch_grid) < 3:
                lanuch_grid += ["1"] * (3 - len(lanuch_grid))
            lanuch_grid = ",".join(lanuch_grid)

            op_dict = {"op_name": op_name, "reset_zero_when_tune": ""}
            op_dict["triton_kernel_args"] = ",".join(modified_arg_exclude_constexpr)
            op_dict["key"] = ",".join(self.key_args)
            # when tunning, we need to reset the out to zero.
            if "reset_zero_when_tune" in other_config.keys():
                op_dict["reset_zero_when_tune"] = other_config["reset_zero_when_tune"]

            paddle_custom_op_file_path = f"{generated_dir}/{op_name}.cu"
            so_path = find_so_path(generated_dir, python_package_name)

            if so_path is None:
                print("== we do not find so_path, we need to compile it")
                with open(paddle_custom_op_file_path, "w") as f:
                    f.write(
                        SubstituteTemplate(
                            self.custom_op_template,
                            op_dict,
                        )
                    )
                    f.close()

                # ahead of time compile command.
                aot_template = (
                    f"""{python_path}   {compile_file} {py_script_file}   -n {func.__name__} -o {generated_dir}/{op_name}_kernel --out-name {op_name}_kernel  """
                    + """ -w {num_warps} -ns {num_stages} """
                    + f""" -s"{address_hint} {value_hint} {const_args}" """
                    + f"""  -g "{lanuch_grid}" """
                )
                all_tune_config = list(self.tune_config)
                if len(all_tune_config) == 0:
                    # when user do not specify config, we use const_hint_dict as config.
                    all_tune_config = [const_hint_dict]
                    # reset const_hint_dict as empty.
                    const_hint_dict = {}
                codegen_commands = []
                for config in all_tune_config:
                    for key in const_hint_dict.keys():
                        if const_hint_dict[key] is not None:
                            if key not in config.keys():
                                config[key] = const_hint_dict[key]
                            else:
                                if config[key] == const_hint_dict[key]:
                                    pass
                                else:
                                    message = f"you specify {key} both in arguments and config, and they are not same, this is wrong."
                                    raise ValueError(message)
                        else:
                            assert key in config.keys(), f"you must specify {key} in your config."
                    if "num_warps" not in config.keys():
                        config["num_warps"] = 4
                    if "num_stages" not in config.keys():
                        config["num_stages"] = 4

                    for key in config:
                        assert config[key] is not None, f"{key} must be specified."
                    codegen_command = aot_template.format(
                        **config,
                    )
                    print(codegen_command)
                    codegen_commands.append(codegen_command)
                multi_process_do(codegen_commands)

                link_command = f"{python_path}  {link_file}  {generated_dir}/*.h -o {generated_dir}/{op_name}_kernel"
                re = os.system(link_command)
                assert re == 0

                # rename the .c file to .cu
                rename_c_to_cu(generated_dir)
                # build the package to so, not install
                build_package(generated_dir, python_package_name)

            if op_name not in OpProtoHolder.instance().op_proto_map.keys():
                so_path = find_so_path(generated_dir, python_package_name)
                print("== we find so_path: ", so_path)
                assert so_path is not None
                paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

        self.decorator = decorator

    def __getitem__(self, op_name_and_grid):
        assert len(op_name_and_grid) >= 3, "len(op_name_and_grid) must >= 3."
        self.op_name = op_name_and_grid[0]
        self.custom_op_template = op_name_and_grid[1]
        self.grid = op_name_and_grid[2]
        if len(op_name_and_grid) == 3:
            self.tune_config = {}
        else:
            self.tune_config = op_name_and_grid[3]

        return self.decorator


def paddle_use_triton(other_config={}, key=[]):
    def decorator(func):
        return KernelInterface(func, other_config, key)

    return decorator
