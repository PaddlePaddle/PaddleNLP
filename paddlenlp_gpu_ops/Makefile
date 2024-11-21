.PHONY: pre-release bdist-wheel

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

bdist-wheel:
	cd csrc && python setup.py build && cd .. && python setup.py bdist_wheel

pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch
