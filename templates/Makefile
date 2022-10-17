.DEFAULT_GOAL := add-new-model

.PHONY: add-new-model
add-new-model:
	cookiecutter ./add-new-model --overwrite-if-exists -o template-results

install:
	pip install -r requirements.txt