# Makefile for PaddleNLP
#
# 	GitHb: https://github.com/PaddlePaddle/PaddleNLP
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

# get source_glob with git tool
export PYTHONPATH=./

file:
	python scripts/run_test.py

SOURCE_GLOB=$(wildcard paddlenlp/**/*.py tests/**/*.py examples/**/*.py model_zoo/**/*.py)

.PHONY: all
all : clean lint

.PHONY: lint
lint: pylint
# lint: pylint mypy

.PHONY: isort
isort:
	isort --atomic ${DIFF_FILES}

# disable: TODO list temporay
.PHONY: pylint
pylint:
	pylint \
		--load-plugins pylint_quotes \
		$(SOURCE_GLOB)

# default for cpu-test
.PHONY: test
test: lint pytest

.PHONY: gpu-test
gpu-test:
	echo "only run gpu-test"

.PHONY: format
format:
	pre-commit run yapf

.PHONY: deploy-ppdiffusers
deploy-ppdiffusers:
	cd ppdiffusers && make

.PHONY: install-ppdiffusers
install-ppdiffusers:
	cd ppdiffusers && make install

