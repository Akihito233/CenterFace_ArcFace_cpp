#1.cmake verson
cmake_minimum_required(VERSION 2.8)


#2.project name
project(demo C CXX)

#opencv4.1.0
find_package(OpenCV REQUIRED)

#3.set environment variable
# set(CMAKE_CXX_FLAGS "-g  -std=c++17 -lstdc++fs")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++17 -lstdc++fs")
set(ONNXRUNTIME_ROOT_PATH /home/space/onnxruntime-linux-x64-gpu-1.10.0)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)

#4.include
include_directories(${ONNXRUNTIME_INCLUDE_DIRS}
	${CMAKE_CURRENT_LIST_DIR}/cpp
	)

#5.source directory
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/cpp/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/cpp/lite/ort/core/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/cpp/lite/ort/cv/*.cpp
	${CMAKE_CURRENT_LIST_DIR}/demo.cpp)

set(TEST_COMPILE_CODE ${TEST_SRC})


add_executable(demo ${TEST_COMPILE_CODE})

target_link_libraries(demo ${OpenCV_LIBS}
	${ONNXRUNTIME_LIB}
	stdc++fs)
