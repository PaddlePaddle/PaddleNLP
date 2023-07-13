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
	git merge-base develop HEAD

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: test
test: unit-test

unit-test:
	PYTHONPATH=$(shell pwd) pytest -v \
		-n auto \
		--durations 20 \
		--cov paddlenlp \
		--cov-report xml:coverage.xml

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
