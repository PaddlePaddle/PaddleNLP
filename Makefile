# Makefile for PaddleNLP
#
# 	GitHb: https://github.com/PaddlePaddle/PaddleNLP
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

.PHONY: all
all : lint test

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 
.PHONY: lint
lint: yapf isort flake8 pylint

.PHONY: yapf
yapf:
	pre-commit run

.PHONY: isort
isort:
	python scripts/run_test.py isort

.PHONY: flake8
flake8:
	python scripts/run_test.py flake8

# disable: TODO list temporay
.PHONY: pylint
pylint:
	python scripts/run_test.py pylint

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 

# default for cpu-test
.PHONY: test
test: 
	example-test 

.PHONY: gpu-test
gpu-test:
	echo "only run gpu-test"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


.PHONY: format
format:
	pre-commit run yapf

.PHONY: deploy-ppdiffusers
deploy-ppdiffusers:
	cd ppdiffusers && make

.PHONY: install-ppdiffusers
install-ppdiffusers:
	cd ppdiffusers && make install

