# Makefile for PaddleNLP
#
# 	GitHb: https://github.com/PaddlePaddle/PaddleNLP
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

.PHONY: all
all : lint test

# # # # # # # # # # # # # # # Format Block # # # # # # # # # # # # # # # 

format:
	$(eval modified_py_files := $(shell python scripts/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		isort $(modified_py_files); \
		yapf --verbose -i $(modified_py_files); \
		git add ${modified_py_files}; \
	else \
		echo "No library .py files were modified"; \
	fi

	# pre-commit run

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 
.PHONY: lint
lint:
	$(eval modified_py_files := $(shell python scripts/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		isort --check-only $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi
	pre-commit run

.PHONY: yapf
yapf:
	pre-commit run

.PHONY: isort
isort:
	black --preview $(modified_py_files); \
	python scripts/run_test.py isort

.PHONY: flake8
flake8:
	python scripts/run_test.py flake8

# disable: TODO list temporay
.PHONY: pylint
pylint:
	python scripts/run_test.py pylint

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

