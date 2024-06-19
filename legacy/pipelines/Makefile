
.DEFAULT_GOAL := all

.PHONY: all
all: deploy-version build deploy

.PHONY: build
build:
	python3 setup.py sdist bdist_wheel

.PHONY: deploy
deploy:
	make deploy-version
	twine upload --skip-existing dist/*

.PHONY: deploy-version
deploy-version:
	echo "VERSION = '$$(cat VERSION)'" > pipelines/version.py

.PHONY: install
install:
	pip install -r requirements.txt


.PHONY: test
test: unit-test

unit-test:
	PYTHONPATH=$(shell pwd) pytest tests

.PHONY: version
version:
	@newVersion=$$(awk -F. '{print $$1"."$$2"."$$3+1}' < VERSION) \
		&& echo $${newVersion} > VERSION \
		&& git add VERSION \
		&& git commit -m "🔥 update version to $${newVersion}" > /dev/null \
		&& echo "Bumped version to $${newVersion}"