
.DEFAULT_GOAL := all

.PHONY: all
all: test build deploy-version deploy

.PHONY: build
build:
	make deploy-version
	python3 setup.py sdist bdist_wheel
	pip install --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple dist/ppdiffusers-0.6.0-py3-none-any.whl

.PHONY: deploy
deploy:
	pip install twine
	twine upload --skip-existing dist/*

.PHONY: deploy-version
deploy-version:
	echo "VERSION = '$$(cat VERSION)'" > ppdiffusers/version.py

install:
	pip install -r requirements.txt

test:
	TWINE_USERNAME=paddle-dev TWINE_PASSWORD="#M9:2VS+FA[;hc?x" make deploy
