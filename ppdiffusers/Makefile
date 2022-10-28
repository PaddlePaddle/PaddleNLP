
.DEFAULT_GOAL := all

.PHONY: all
all: test build deploy-version deploy

.PHONY: build
build:
	python3 setup.py sdist bdist_wheel

.PHONY: deploy
deploy:
	twine upload --skip-existing dist/*

.PHONY: deploy-version
deploy-version:
	echo "VERSION = '$$(cat VERSION)'" > ppdiffusers/version.py

install:
	pip install -r requirements.txt

test:
	echo "to enable the testing in ppdiffusers"