import os
import sys, shutil
import custom_build_tools.build as build_tools
from setuptools import setup

REQUIRED_PACKAGES = [
    'setuptools >= 39.2.0',
    'cmake >= 3.11.0',
    'cffi >= 1.11.5',
    'mkl >= 2018.0.0',
    'pyyaml >= 3.12',
    'numpy >= 1.14.0',
]

# setup(
#     name='pytorch-gpu-macosx',
#     version='0.5.0',
#     description='Unoffcial NVIDIA CUDA GPU support version of PyTorch for MAC OSX 10.13',
#     author='Carl Cheung',
#     author_email='zylo117@hotmail.com',
#     url='https://github.com/zylo117/pytorch',
#     install_requires=REQUIRED_PACKAGES,
#     keywords='gpu cuda torch tensor machine learning', )


pytorch_src_path = '../'

# modify CMakeLists
if not os.path.exists(pytorch_src_path + 'CMakeLists_bakcup.txt'):
    shutil.copy(pytorch_src_path + 'CMakeLists.txt', pytorch_src_path + 'CMakeLists_bakcup.txt')
    cmakelists = open(pytorch_src_path + 'CMakeLists.txt', 'r')
    cml_data = cmakelists.readlines()
    for i, l in enumerate(cml_data):
        if 'CMAKE_RUNTIME_OUTPUT_DIRECTORY' in l:
            cml_data[i] = cml_data[
                              i] + '\n' + 'if(APPLE)\n\tset(CMAKE_FIND_FRAMEWORK LAST)\n\tset(CMAKE_FIND_APPBUNDLE LAST)\nendif()\n'
            break
    cmakelists = open('../CMakeLists.txt', 'w')
    cmakelists.writelines(cml_data)
    cmakelists.close()

pwd = sys.path[0]

# build PT
python_exec = build_tools.get_python_path()
python_bin = '/'.join(build_tools.get_python_path().split('/')[:-1])
python_lib_path = build_tools.get_python_package()

build_tools.git_clone()

print('[INFO] Building PyTorch...')
print("[INFO] It's going to last for about 20 minutes on Intel i7-6700")
print('[INFO] Build Complete')
