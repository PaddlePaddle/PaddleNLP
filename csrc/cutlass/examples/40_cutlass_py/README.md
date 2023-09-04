# CUTLASS Python Interface Examples
This directory contains examples of using CUTLASS's Python interface. It consists of two types of examples:
* _Basic examples_: minimal examples that illustrate how to set up GEMMs, convolutions, and grouped GEMM operations
* [_Customizable examples_](customizable): examples that allow one to specify a variety of template parameters for the given kernel

## Setting up the Python interface
Please follow the instructions [here](/tools/library/scripts/pycutlass/README.md#installation) to set up the Python API.

## Running examples
Each of the basic examples can be run as follows:
```shell
# Run the GEMM example
python gemm.py

# Run the Conv2d example
python conv2d.py

# Run the grouped GEMM example
python gemm_grouped.py
```

To run the customizable examples, refer to the README in the [customizable](customizable) directory.
