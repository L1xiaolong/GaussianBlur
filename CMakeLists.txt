cmake_minimum_required(VERSION 3.2)
project(gaussianFilterTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

#find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${OpenMP_CXX_FLAGS}")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 ${OpenMP_C_FLAGS}")
set(CMAKE_BUILD_TYPE release)

set(CMAKE_OSX_ARCHITECTURES  "arm64")
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gaussianFilterTest main.cpp gaussFilter.c)
target_link_libraries(gaussianFilterTest ${OpenCV_LIBS})
