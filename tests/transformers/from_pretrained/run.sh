set -x
export HF_ENDPOINT=https://hf-mirror.com
PYTHONPATH=../../../:$PYTHONPATH \
python3 test_image_processor.py