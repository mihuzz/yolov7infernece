cmake_minimum_required(VERSION 3.5)

project(yoloTRT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/usr/local/TensorRT-8.4.1.5/include/)
link_directories(/usr/local/TensorRT-8.4.1.5/lib/)
set(TENSORRT_ROOT /usr/local/TensorRT-8.4.1.5)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
set(LIB ${LIB} ${TENSORRT_LIBRARY})
set(INC ${INC} ${TENSORRT_INCLUDE_DIR})


include_directories(${PROJECT_SOURCE_DIR}/include)

INCLUDE_DIRECTORIES(
     ${CMAKE_CURRENT_SOURCE_DIR}/include
 )


SET(PROC_ALL_FILES ${SRCS})
file(GLOB SOURCE_FILES src/*.cpp)
add_executable(yoloTRT ${PROJECT_SOURCE_DIR}/main.cpp yolo.cpp)

target_link_libraries(yoloTRT ${OpenCV_LIBRARIES})
target_link_libraries(yoloTRT "${TensorRT-8.4.1.5}")
target_link_libraries(yoloTRT nvonnxparser)
target_link_libraries(yoloTRT nvinfer)
target_link_libraries(yoloTRT nvinfer_plugin)
target_link_libraries(yoloTRT cudart)
add_definitions(-O2 -pthread)
