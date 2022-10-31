
.DEFAULT_GOAL := all

.PHONY: all
all: build deploy-version deploy

.PHONY: build
build:
	python3 setup.py sdist bdist_wheel

.PHONY: deploy
deploy:
	make deploy-version
	twine upload --skip-existing dist/*

.PHONY: deploy-version
deploy-version:
	echo "VERSION = '$$(cat VERSION)'" > ppdiffusers/version.py

.PHONY: install
install:
	pip install -r requirements.txt
