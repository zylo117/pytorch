1. copy this to root CMakeLists.txt on about line 163

# because stupid CMakeLists considers "/usr/local/cuda/lib/stubs/cuda.framework" is cuda lib path instead of "/usr/local/cuda/lib"

if(APPLE)
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)
endif()

2. conda install numpy pyyaml setuptools cmake cffi mkl

3. git clone --recursive https://github.com/pytorch/pytorch

4. cd pytorch

5. Install XCode/Command line tools 8.2, and then run this "sudo xcode-select --switch /Library/Developer/CommandLineTools"

5. if you failed building before, use "MACOSX_DEPLOYMENT_TARGET=10.9 CC=gcc CXX=g++ python setup.py clean" first

6. MACOSX_DEPLOYMENT_TARGET=10.9 CC=gcc CXX=g++ python setup.py install

# Noticing there are gcc&g++ instead of clang&clang++(PyTorch offcial solution, which doesn't work for me, but you're welcome to try if the other fails), or you might fail

7. (Optional) Reinstall XCode/Command line tools newest version and then run this "sudo xcode-select --switch /Library/Developer/CommandLineTools", or homebrew might not work