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
	chmod +x scripts/paddle_log
	scripts/paddle_log
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
	chmod +x scripts/paddle_log
	scripts/paddle_log
	PYTHONPATH=$(shell pwd) pytest -v \
		-n auto \
		--retries 1 --retry-delay 1 \
		--durations 20 \
		--cov paddlenlp \
		--cov-report xml:coverage.xml

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.PHONY: install
install:
	chmod +x scripts/paddle_log
	scripts/paddle_log
	pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
	pip install -r requirements-dev.txt
	pip install -r requirements.txt
	pip install -r paddlenlp/experimental/autonlp/requirements.txt
	pre-commit install


.PHONY: deploy-ppdiffusers
deploy-ppdiffusers:
	chmod +x scripts/paddle_log
	scripts/paddle_log
	cd ppdiffusers && make install && make

.PHONY: deploy-paddle-pipelines
deploy-paddle-pipelines:
	chmod +x scripts/paddle_log
	scripts/paddle_log
	cd pipelines && make install && make

.PHONY: deploy-paddlenlp
deploy-paddlenlp:
	chmod +x scripts/paddle_log
	scripts/paddle_log
	# install related package
	make install
	# build
	python3 setup.py sdist bdist_wheel
	# upload
	twine upload --skip-existing dist/*

.PHONY: regression-all
release: 
	chmod +x scripts/paddle_log
	scripts/paddle_log
	bash ./scripts/regression/run_release.sh 0 0,1 all

.PHONY: regression-key
key: 
	chmod +x scripts/paddle_log
	scripts/paddle_log
	bash ./scripts/regression/run_release.sh 0 0,1 p0
