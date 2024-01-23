from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="exllamav2_kernels",
    ext_modules=[
        CUDAExtension(
            name="exllamav2_kernels",
            sources=[
                "exllamav2_kernels/ext.cpp",
                "exllamav2_kernels/cuda/h_gemm.cu",
                "exllamav2_kernels/cuda/lora.cu",
                "exllamav2_kernels/cuda/pack_tensor.cu",
                "exllamav2_kernels/cuda/quantize.cu",
                "exllamav2_kernels/cuda/q_matrix.cu",
                "exllamav2_kernels/cuda/q_attn.cu",
                "exllamav2_kernels/cuda/q_mlp.cu",
                "exllamav2_kernels/cuda/q_gemm.cu",
                "exllamav2_kernels/cuda/rms_norm.cu",
                "exllamav2_kernels/cuda/rope.cu",
                "exllamav2_kernels/cuda/cache.cu",
                "exllamav2_kernels/cpp/quantize_func.cpp",
                "exllamav2_kernels/cpp/sampling.cpp"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
