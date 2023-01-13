# Makefile for PaddleNLP
#
# 	GitHb: https://github.com/PaddlePaddle/PaddleNLP
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

.PHONY: all
all : lint test
check_dirs := applications examples model_zoo paddlenlp pipelines ppdiffusers scripts tests 
# # # # # # # # # # # # # # # Format Block # # # # # # # # # # # # # # # 

format:
	pre-commit run isort
	pre-commit run black

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 

.PHONY: lint
lint:
	$(eval modified_py_files := $(shell python scripts/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo ${modified_py_files}; \
		pre-commit run --files ${modified_py_files}; \
	else \
		echo "No library .py files were modified"; \
	fi	

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: test
test: unit-test

unit-test:
	PYTHONPATH=$(shell pwd) pytest -v \
		-n auto \
		--cov paddlenlp \
		--cov-report xml:coverage.xml

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # FastTokenizer Block # # # # # # # # # # #

.PHONY: fast_tokenizer_cpp_compile

fast_tokenizer_cpp_compile:
	cd fast_tokenizer && mkdir -p build_cpp && cd build_cpp && \
	cmake .. -DWITH_PYTHON=OFF -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release && \
	make -j4 && cd ../..

fast_tokenizer_cpp_test:
	cd fast_tokenizer/build_cpp/fast_tokenizer/test && \
	bash ../../../run_fast_tokenizer_cpp_test.sh

fast_tokenizer_python_compile:
	pip install numpy wheel && \
	cd fast_tokenizer && mkdir -p build_py && cd build_py && \
	cmake .. -DWITH_PYTHON=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release && \
	make -j4 && pip install dist/*whl && cd ../..

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.PHONY: install
install:
	pip install -r requirements-dev.txt
	pip install -r requirements.txt
	pip install -r paddlenlp/experimental/autonlp/requirements.txt
	pre-commit install


.PHONY: deploy-ppdiffusers
deploy-ppdiffusers:
	cd ppdiffusers && make install && make

.PHONY: deploy-paddle-pipelines
deploy-paddle-pipelines:
	cd pipelines && make install && make

.PHONY: deploy-paddlenlp
deploy-paddlenlp:
	# install related package
	make install
	# build
	python3 setup.py sdist bdist_wheel
	# upload
	twine upload --skip-existing dist/*

.PHONY: regression-all
release: 
	bash ./scripts/regression/run_release.sh 0 0,1 all

.PHONY: regression-key
key: 
	bash ./scripts/regression/run_release.sh 0 0,1 p0
