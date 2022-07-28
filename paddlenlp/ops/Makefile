# 
# Makefile for Manually building faster transformer
# 

.DEFAULT_GOAL := build-ft


.PYONH: build-ft
build-ft:
	echo "clean building cache"
	rm -rf build && mkdir -p build

	echo "star to building faster transformer"
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python && make -j
	