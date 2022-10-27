# Makefile for PaddleNLP
#
# 	GitHb: https://github.com/PaddlePaddle/PaddleNLP
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

# get source_glob with git tool

SOURCE_GLOB=$(wildcard paddlenlp/**/*.py tests/**/*.py examples/**/*.py model_zoo/**/*.py)

IGNORE_PEP=E203,E221,E241,E272,E501,F811


.PHONY: all
all : clean lint

.PHONY: clean
clean:
	rm -fr dist/*

.PHONY: lint
lint: pylint pycodestyle flake8 mypy


# disable: TODO list temporay
.PHONY: pylint
pylint:
	pylint \
		--load-plugins pylint_quotes \
		--disable=W0511,R0801,cyclic-import,C4001 \
		$(SOURCE_GLOB)

.PHONY: pycodestyle
pycodestyle:
	pycodestyle \
		--statistics \
		--count \
		--ignore="${IGNORE_PEP}" \
		$(SOURCE_GLOB)

.PHONY: flake8
flake8:
	flake8 \
		--ignore="${IGNORE_PEP}" \
		$(SOURCE_GLOB)

.PHONY: mypy
mypy:
	MYPYPATH=stubs/ mypy \
		$(SOURCE_GLOB)

.PHONY: uninstall-git-hook
uninstall-git-hook:
	pre-commit clean
	pre-commit gc
	pre-commit uninstall
	pre-commit uninstall --hook-type pre-push

.PHONY: install-git-hook
install-git-hook:
	# cleanup existing pre-commit configuration (if any)
	pre-commit clean
	pre-commit gc
	# setup pre-commit
	# Ensures pre-commit hooks point to latest versions
	pre-commit autoupdate
	pre-commit install
	pre-commit install --hook-type pre-push

.PHONY: deploy-install
deploy-install:
		

.PHONY: install
install:
	pip3 install -r requirements.txt
	pip3 install -r paddlenlp/cli/requirements.txt
	pip3 install -r test/requirements.txt
	pip3 install -r docs/requirements.txt
	$(MAKE) install-git-hook

.PHONY: unittest 
pytest:
	unittest tests/

.PHONY: test-unit
test-unit: pytest

.PHONY: test
test: lint pytest

.PHONY: format
format:
	pre-commit run yapf

.PHONY: dist
dist:
	python3 setup.py sdist bdist_wheel

.PHONY: publish
publish:
	PATH=~/.local/bin:${PATH} twine upload dist/*
