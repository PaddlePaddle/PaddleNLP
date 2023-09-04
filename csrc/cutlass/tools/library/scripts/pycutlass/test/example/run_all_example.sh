#################################################################################################
#
# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

pushd $CUTLASS_PATH/examples/40_cutlass_py/customizable

python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 32 32 16 -s 4 -w 2 2 1 -cc 80 -la ColumnMajor -aa 1 -lb RowMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1

python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 64 64 16 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb ColumnMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 2

python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_bf16 -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1

python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm GemmSplitKParallel -k 2

python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_f32 -op TensorOp -b 64 64 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 4

python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb RowMajor -ab 8 -lc ColumnMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle4 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1

python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 2

python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 256 128 64 -s 3 -w 4 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm GemmSplitKParallel -k 3

python gemm.py -i 16 8 16 -ta bfloat16 -tb bfloat16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 64 128 64 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm GemmSplitKParallel -k 5

python gemm.py -i 16 8 32 -ta int8 -tb int8 -tc int8 -tacc int32 -m multiply_add -op TensorOp -b 128 128 128 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 16 -lb ColumnMajor -ab 16 -lc RowMajor -ac 16 -te float32 -ep FastLinearCombinationClamp -sw IdentitySwizzle2 -p 512 512 512 -alpha 1.0 -beta 0.0 -gm Gemm -k 1

python gemm_grouped.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc ColumnMajor -ac 4 -te float32 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 1.0 -beta 0.0 -pm Device

python gemm_grouped.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 64 64 16 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb RowMajor -ab 1 -lc ColumnMajor -ac 1 -te float64 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 1.0 -beta 1.0 -pm Host

python gemm_grouped.py -i 1 1 1 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op Simt -b 128 64 8 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb RowMajor -ab 1 -lc RowMajor -ac 1 -te float32 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 2.0 -beta 1.0 -pm Device

python gemm_grouped.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc ColumnMajor -ac 4 -te float32 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 2.0 -beta 1.0 -pm Device

python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 16 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co fprop -st Strided -ia optimized -sm Serial -k 1 -nhwc 1 13 17 8 -krsc 24 3 3 8 -pad 0 0 0 0 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 0.0

python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 16 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 2 -lb TensorNHWC -ab 2 -lc TensorNHWC -ac 2 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -co fprop -st Strided -ia optimized -sm Serial -k 2 -nhwc 1 4 4 12 -krsc 8 3 3 12 -pad 0 0 0 0 -stride 3 3 -dilation 1 1 -alpha 1.0 -beta 1.0

python conv2d.py -i 1 1 1 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op Simt -b 128 128 8 -s 4 -w 4 2 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 1 -te float32 -ep LinearCombination -sw IdentitySwizzle4 -co fprop -st Strided -ia analytic -sm Parallel -k 3 -nhwc 1 71 80 32 -krsc 64 5 5 32 -pad 2 2 2 2 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 1.0

python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 1 -lb TensorNHWC -ab 1 -lc TensorNHWC -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co wgrad -st Strided -ia optimized -sm Serial -k 1 -nhwc 1 8 8 1 -krsc 1 3 3 1 -pad 1 1 1 1 -stride 1 1 -dilation 1 1 -alpha 1.0 -beta 0.0

python conv2d.py -i 1 1 1 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op Simt -b 128 128 8 -s 4 -w 2 4 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 1 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co wgrad -st Strided -ia optimized -sm Serial -k 2 -nhwc 1 27 27 256 -krsc 512 3 3 256 -pad 1 1 1 1 -stride 2 1 -dilation 1 1 -alpha 1.0 -beta 0.0

python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 16 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 4 -te float32 -ep LinearCombination -sw StridedDgradIdentitySwizzle1 -co dgrad -st Strided -ia optimized -sm Serial -k 2 -nhwc 1 27 27 256 -krsc 512 3 3 256 -pad 1 1 1 1 -stride 2 1 -dilation 1 1 -alpha 1.0 -beta 0.0

python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 8 -lb TensorNHWC -ab 8 -lc TensorNHWC -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co fprop -st Strided -ia optimized -sm Serial -k 1 -nhwc 1 27 27 256 -krsc 512 3 3 256 -pad 1 1 1 1 -stride 2 1 -dilation 1 1 -alpha 1.0 -beta 0.0

python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 2 -lb TensorNHWC -ab 2 -lc TensorNHWC -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co fprop -st Strided -ia few_channels -sm Serial -k 1 -nhwc 1 16 16 2 -krsc 16 3 3 2 -pad 1 1 1 1 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 0.0

python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 8 -lb TensorNHWC -ab 8 -lc TensorNHWC -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -co fprop -st Strided -ia fixed_channels -sm Serial -k 1 -nhwc 1 8 8 8 -krsc 16 3 3 8 -pad 1 1 1 1 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 0.0

python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 4 -te float32 -ep LinearCombination -sw StridedDgradIdentitySwizzle1 -co dgrad -st Strided -ia optimized -sm Serial -k 1 -nhwc 1 56 56 12 -krsc 8 1 1 12 -pad 0 0 0 0 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 0.0

python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 32 32 16 -s 4 -w 2 2 1 -cc 80 -la ColumnMajor -aa 1 -lb RowMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1 -bias -activ relu

python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 64 64 16 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb ColumnMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 2 -bias -activ leaky_relu -activ_arg 0.2

python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm GemmSplitKParallel -k 2 -bias -activ tanh

python gemm_grouped.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 64 64 16 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb RowMajor -ab 1 -lc ColumnMajor -ac 1 -te float64 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 0.0 -beta 0.5 -pm Host -bias -activ sigmoid -bias -activ sigmoid

python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 16 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 2 -lb TensorNHWC -ab 2 -lc TensorNHWC -ac 2 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -co fprop -st Strided -ia optimized -sm Serial -k 2 -nhwc 1 4 4 12 -krsc 8 3 3 12 -pad 0 0 0 0 -stride 3 3 -dilation 1 1 -alpha 0.0 -beta 0.5 -bias -activ silu

python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 2 -lb TensorNHWC -ab 2 -lc TensorNHWC -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co fprop -st Strided -ia few_channels -sm Serial -k 1 -nhwc 1 16 16 2 -krsc 16 3 3 2 -pad 1 1 1 1 -stride 2 2 -dilation 1 1 -alpha 0.0 -beta 0.5 -bias -activ hardswish

python gemm.py -i 16 8 16 -ta bfloat16 -tb bfloat16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 64 128 64 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -p 512 256 128 -alpha 0.0 -beta 0.5 -gm GemmSplitKParallel -k 5 -bias -activ gelu

python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_bf16 -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw BatchedIdentitySwizzle -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Batched -k 1 -batch 3

python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb RowMajor -ab 8 -lc ColumnMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle4 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Array -k 1 -batch 2

python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_bf16 -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -epv RowBroadcast -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1

python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 32 32 16 -s 4 -w 2 2 1 -cc 80 -la ColumnMajor -aa 1 -lb RowMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -epv ColumnBroadcast -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1

python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb RowMajor -ab 8 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -epv RowReduction -sw IdentitySwizzle4 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1

python gemm.py -i 16 8 16 -ta bfloat16 -tb bfloat16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 64 128 64 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -epv ColumnReduction -sw IdentitySwizzle2 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1

python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_bf16 -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -epv RowReduction -sw BatchedIdentitySwizzle -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Batched -k 1 -batch 3

python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_bf16 -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -epv ColumnBroadcast -sw BatchedIdentitySwizzle -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Array -k 1 -batch 3
popd 
