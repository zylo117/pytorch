from setuptools import setup, find_packages, Extension

REQUIRED_PACKAGES = [
    'setuptools >= 39.2.0',
    'cmake >= 3.11.0',
    'cffi >= 1.11.5',
    'mkl >= 2018.0.0',
    'pyyaml >= 3.12',
    'numpy >= 1.14.0',
]

packages = find_packages(exclude=('tools', 'tools.*', 'caffe2', 'caffe2.*', 'caffe', 'caffe.*'))

setup(
    name='pytorch-gpu-macosx',
    version='0.5.0',
    description='Unoffcial NVIDIA CUDA GPU support version of PyTorch for MAC OSX 10.13',
    author='Carl Cheung',
    author_email='zylo117@hotmail.com',
    url='https://github.com/zylo117/pytorch',
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    package_data={
    'torch': [
        'lib/*.so*',
        'lib/*.dylib*',
        'lib/*.dll',
        'lib/*.lib',
        'lib/torch_shm_manager',
        'lib/*.h',
        'lib/include/ATen/*.h',
        'lib/include/ATen/detail/*.h',
        'lib/include/ATen/cuda/*.h',
        'lib/include/ATen/cuda/*.cuh',
        'lib/include/ATen/cuda/detail/*.h',
        'lib/include/ATen/cudnn/*.h',
        'lib/include/ATen/cuda/detail/*.cuh',
        'lib/include/pybind11/*.h',
        'lib/include/pybind11/detail/*.h',
        'lib/include/TH/*.h',
        'lib/include/TH/generic/*.h',
        'lib/include/THC/*.h',
        'lib/include/THC/*.cuh',
        'lib/include/THC/generic/*.h',
        'lib/include/THCUNN/*.cuh',
        'lib/include/torch/csrc/*.h',
        'lib/include/torch/csrc/autograd/*.h',
        'lib/include/torch/csrc/jit/*.h',
        'lib/include/torch/csrc/utils/*.h',
        'lib/include/torch/csrc/cuda/*.h',
        'lib/include/torch/torch.h',
    ]},
    keywords='gpu cuda torch tensor machine learning', )