from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_setup_ops',
    ext_modules=CUDAExtension(
        sources=[
            'neox_rope.cu']
    )
)