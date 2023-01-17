# Makefile for fast_tokenizer
#
# 	GitHb: https://github.com/PaddlePaddle/PaddleNLP
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

# Compile and test for fast_tokenizer cpp library

.PHONY: fast_tokenizer_cpp_compile

fast_tokenizer_cpp_compile:
	mkdir -p build_cpp && cd build_cpp && \
	cmake .. -DWITH_PYTHON=OFF -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release && \
	make -j4

.PHONY: fast_tokenizer_cpp_test

fast_tokenizer_cpp_test:
	bash run_fast_tokenizer_cpp_test.sh build_cpp/fast_tokenizer/test

# Compile and test for fast_tokenizer python library

.PHONY: fast_tokenizer_python_install

fast_tokenizer_python_install:
	pip install numpy wheel pytest paddlepaddle ..

.PHONY: fast_tokenizer_python_compile

fast_tokenizer_python_compile:
	mkdir -p build_py && cd build_py && \
	cmake .. -DWITH_PYTHON=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release && \
	make -j4

.PHONY: fast_tokenizer_python_test

fast_tokenizer_python_test:
	pip install build_py/dist/*whl && pytest build_py/python/tests
