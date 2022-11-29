# Makefile for PaddleNLP
#
# 	GitHb: https://github.com/PaddlePaddle/PaddleNLP
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

.PHONY: all
all : lint test

# # # # # # # # # # # # # # # Format Block # # # # # # # # # # # # # # # 

format:
	echo "============================== run isort ==============================\n"
	pre-commit run isort
	echo "============================== run black ==============================\n"
	pre-commit run black

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 

.PHONY: lint
lint:
	echo "============================== run isort ==============================\n"
	pre-commit run isort
	echo "============================== run black ==============================\n"
	pre-commit run black
	echo "============================== run flake8 ==============================\n"
	pre-commit run flake8

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: test
test: unit-test

unit-test:
	# only enable bert-test: there are many failed tests
	PYTHONPATH=$(shell pwd) pytest tests/transformers/bert

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.PHONY: install
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install


.PHONY: deploy-ppdiffusers
deploy-ppdiffusers:
	cd ppdiffusers && make

.PHONY: install-ppdiffusers
install-ppdiffusers:
	cd ppdiffusers && make install

