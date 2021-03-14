cmake_minimum_required(VERSION 3.2)
project(VIGILANT)
set(Torch_DIR /home/alan/Downloads/libtorch/share/cmake/Torch )
set(OPENCV_EXTRA_MODULES_PATH /home/alan/opencv/opencv_contrib-master/modules)
# The following directories is for ARM64 (NVIDIA JETSON NANO)
# set(CMAKE_PREFIX_PATH "/home/jetsonalan/.local/lib/python3.6/site-packages/torch" )
# set(OPENCV_EXTRA_MODULES_PATH /home/jetsonalan/opencv_build/opencv_contrib/modules)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package( OpenCV REQUIRED )
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "CUDA TOOLKIT ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable(main src/main.cpp
              include/camera.h
              src/camera.cpp  )

target_include_directories(main PRIVATE include)

target_link_libraries( main ${OpenCV_LIBS})
target_link_libraries(main  "${TORCH_LIBRARIES}")

set_property(TARGET main PROPERTY CXX_STANDARD 14)



