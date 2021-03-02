cmake_minimum_required(VERSION 2.6)
project(deepsort)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)
link_directories(
    /usr/local/cuda/lib64
)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)

# ===== deepsort =====
add_library(deepsort SHARED ${SRC_DIR})
target_link_libraries(deepsort 
    ${CUDA_LIBS} ${OpenCV_LIBS} 
    cudart nvinfer nvonnxparser
)

# ===== onnx2engine =====
add_executable(onnx2engine ${PROJECT_SOURCE_DIR}/onnx2engine.cpp)
target_link_libraries(onnx2engine
    ${CUDA_LIBS}
    cudart nvinfer nvonnxparser deepsort
)

# ===== demo =====
add_executable(demo ${PROJECT_SOURCE_DIR}/demo.cpp)
target_link_libraries(demo 
    ${CUDA_LIBS} ${OpenCV_LIBS} 
    cudart nvinfer nvonnxparser deepsort
)


