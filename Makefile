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


# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: all-test
all-test: test gpu-test

# default for cpu-test
.PHONY: test
test:
	python -m unittest discover -f -p "test_*.py"

.PHONY: gpu-test
gpu-test:
	# TODO(wj-Mcat): need to intergrate this test into iPipe
	echo "only run gpu-test"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


.PHONY: deploy-ppdiffusers
deploy-ppdiffusers:
	cd ppdiffusers && make

.PHONY: install-ppdiffusers
install-ppdiffusers:
	cd ppdiffusers && make install

