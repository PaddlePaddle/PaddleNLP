# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import subprocess


def is_cuda():
    try:
        import paddle

        return paddle.is_compiled_with_cuda()
    except Exception:
        return False


def get_ext_and_cmd():
    ext_modules = []
    cmdclass = {}
    if is_cuda():
        import shutil

        import paddle
        from paddle.utils.cpp_extension import BuildExtension, CUDAExtension
        from paddle.utils.cpp_extension.cpp_extension import BuildCommand
        from paddle.utils.cpp_extension.extension_utils import custom_write_stub

        custom_ops_path = "./csrc"
        name = "paddlenlp.custom_ops._C"

        class GenerateBuildExtension(BuildExtension):
            """
            Inherited from paddle.utils.cpp_extension.BuildExtension to
            auto generate the python api for custom ops.
            """

            def build_extensions(self) -> None:
                super().build_extensions()
                so_path = self.get_ext_fullpath(self.extensions[0]._full_name)
                # Get .so path and .so suffix
                file_path, ext_suffix = os.path.splitext(so_path)

                # Generate python api file
                python_api_path = file_path + ".py"
                custom_write_stub(so_path.split("/")[-1], python_api_path)

                # Avoid the conflict between the .so file and the .py file
                # when import this package
                new_so_path = file_path + "_pd_" + ext_suffix
                if not os.path.exists(new_so_path):
                    os.rename(rf"{so_path}", rf"{new_so_path}")
                assert os.path.exists(new_so_path)

        def update_git_submodule():
            try:
                subprocess.run(["git", "submodule", "update", "--init"], check=True, cwd=custom_ops_path)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while updating git submodule: {str(e)}")
                raise

        def find_end_files(directory, end_str):
            gen_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(end_str):
                        gen_files.append(os.path.join(root, file))
            return gen_files

        def get_sm_version():
            prop = paddle.device.cuda.get_device_properties()
            cc = prop.major * 10 + prop.minor
            return cc

        def strtobool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ("yes", "true", "t", "y", "1"):
                return True
            elif v.lower() in ("no", "false", "f", "n", "0"):
                return False
            else:
                raise ValueError(
                    f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
                )

        def get_gencode_flags():
            if not strtobool(os.getenv("FLAG_LLM_PDC", "False")):
                cc = get_sm_version()
                if cc == 90:
                    cc = f"{cc}a"
                return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
            else:
                # support more cuda archs
                return [
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-gencode",
                    "arch=compute_75,code=sm_75",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                ]

        gencode_flags = get_gencode_flags()
        library_path = os.environ.get("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")

        sources = [
            f"{custom_ops_path}/gpu/save_with_output.cc",
            f"{custom_ops_path}/gpu/set_value_by_flags.cu",
            f"{custom_ops_path}/gpu/token_penalty_multi_scores.cu",
            f"{custom_ops_path}/gpu/token_penalty_multi_scores_v2.cu",
            f"{custom_ops_path}/gpu/stop_generation_multi_ends.cu",
            f"{custom_ops_path}/gpu/fused_get_rope.cu",
            f"{custom_ops_path}/gpu/get_padding_offset.cu",
            f"{custom_ops_path}/gpu/qkv_transpose_split.cu",
            f"{custom_ops_path}/gpu/rebuild_padding.cu",
            f"{custom_ops_path}/gpu/transpose_removing_padding.cu",
            f"{custom_ops_path}/gpu/write_cache_kv.cu",
            f"{custom_ops_path}/gpu/encode_rotary_qk.cu",
            f"{custom_ops_path}/gpu/get_padding_offset_v2.cu",
            f"{custom_ops_path}/gpu/rebuild_padding_v2.cu",
            f"{custom_ops_path}/gpu/set_value_by_flags_v2.cu",
            f"{custom_ops_path}/gpu/stop_generation_multi_ends_v2.cu",
            f"{custom_ops_path}/gpu/update_inputs.cu",
            f"{custom_ops_path}/gpu/get_output.cc",
            f"{custom_ops_path}/gpu/save_with_output_msg.cc",
            f"{custom_ops_path}/gpu/write_int8_cache_kv.cu",
            f"{custom_ops_path}/gpu/step.cu",
            f"{custom_ops_path}/gpu/quant_int8.cu",
            f"{custom_ops_path}/gpu/dequant_int8.cu",
            f"{custom_ops_path}/gpu/flash_attn_bwd.cc",
            f"{custom_ops_path}/gpu/tune_cublaslt_gemm.cu",
            f"{custom_ops_path}/gpu/sample_kernels/top_p_sampling_reject.cu",
            f"{custom_ops_path}/gpu/update_inputs_v2.cu",
            f"{custom_ops_path}/gpu/set_preids_token_penalty_multi_scores.cu",
            f"{custom_ops_path}/gpu/speculate_decoding_kernels/ngram_match.cc",
            f"{custom_ops_path}/gpu/speculate_decoding_kernels/speculate_save_output.cc",
            f"{custom_ops_path}/gpu/speculate_decoding_kernels/speculate_get_output.cc",
        ]
        sources += find_end_files(f"{custom_ops_path}/gpu/speculate_decoding_kernels", ".cu")

        nvcc_compile_args = gencode_flags
        update_git_submodule()
        nvcc_compile_args += [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            f"-I{custom_ops_path}/gpu",
            f"-I{custom_ops_path}/gpu/cutlass_kernels",
            f"-I{custom_ops_path}/gpu/fp8_gemm_with_cutlass",
            f"-I{custom_ops_path}/gpu/cutlass_kernels/fp8_gemm_fused/autogen",
            f"-I{custom_ops_path}/third_party/cutlass/include",
            f"-I{custom_ops_path}/third_party/cutlass/tools/util/include",
            f"-I{custom_ops_path}/third_party/nlohmann_json/single_include",
            f"-I{custom_ops_path}/gpu/sample_kernels",
        ]

        cc = get_sm_version()
        cuda_version = float(paddle.version.cuda())

        if cc >= 80:
            sources += [f"{custom_ops_path}/gpu/int8_gemm_with_cutlass/gemm_dequant.cu"]

            sources += [
                f"{custom_ops_path}/gpu/append_attention.cu",
                f"{custom_ops_path}/gpu/append_attn/get_block_shape_and_split_kv_block.cu",
                f"{custom_ops_path}/gpu/append_attn/decoder_write_cache_with_rope_kernel.cu",
                f"{custom_ops_path}/gpu/append_attn/speculate_write_cache_with_rope_kernel.cu",
            ]
            sources += find_end_files(f"{custom_ops_path}/gpu/append_attn/template_instantiation", ".cu")

        fp8_auto_gen_directory = f"{custom_ops_path}/gpu/cutlass_kernels/fp8_gemm_fused/autogen"
        if os.path.isdir(fp8_auto_gen_directory):
            shutil.rmtree(fp8_auto_gen_directory)

        if cc == 89 and cuda_version >= 12.4:
            os.system(f"python {custom_ops_path}/utils/auto_gen_fp8_fp8_gemm_fused_kernels.py --cuda_arch 89")
            os.system(f"python {custom_ops_path}/utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels.py --cuda_arch 89")
            sources += find_end_files(fp8_auto_gen_directory, ".cu")
            sources += [
                f"{custom_ops_path}/gpu/fp8_gemm_with_cutlass/fp8_fp8_half_gemm.cu",
                f"{custom_ops_path}/gpu/fp8_gemm_with_cutlass/fp8_fp8_half_cuda_core_gemm.cu",
                f"{custom_ops_path}/gpu/fp8_gemm_with_cutlass/fp8_fp8_fp8_dual_gemm.cu",
            ]

        if cc >= 90 and cuda_version >= 12.0:
            nvcc_compile_args += ["-DNDEBUG"]
            os.system(f"python {custom_ops_path}utils/auto_gen_fp8_fp8_gemm_fused_kernels_sm90.py --cuda_arch 90")
            os.system(f"python {custom_ops_path}utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels_sm90.py --cuda_arch 90")
            sources += find_end_files(fp8_auto_gen_directory, ".cu")
            sources += [
                f"{custom_ops_path}/gpu/fp8_gemm_with_cutlass/fp8_fp8_half_gemm.cu",
                f"{custom_ops_path}/gpu/fp8_gemm_with_cutlass/fp8_fp8_half_cuda_core_gemm.cu",
                f"{custom_ops_path}/gpu/fp8_gemm_with_cutlass/fp8_fp8_fp8_dual_gemm.cu",
            ]

        cuda_module = CUDAExtension(
            sources=sources,
            extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_compile_args},
            libraries=["cublasLt"],
            library_dirs=[library_path],
        )
        cuda_module.name = name
        ext_modules.append(cuda_module)

        cmdclass["build_ext"] = GenerateBuildExtension.with_options(no_python_abi_suffix=True)
        build_base = os.path.join("build", name)
        cmdclass["build"] = BuildCommand.with_options(build_base=build_base)

    return ext_modules, cmdclass
