cmake_minimum_required(VERSION 2.8)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# if GCC > 5
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS}")
endif()

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(lambdarank SHARED lambdarank.cc)