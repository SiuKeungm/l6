cmake_minimum_required(VERSION 3.8)
project(l5)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -o2 -pthread")
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /usr/include/eigen3
        ${Sophus_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS})

add_executable(optical optical_flow.cpp)
target_link_libraries(optical ${OpenCV_LIBS})

add_executable(direct direct_method.cpp)
target_link_libraries(direct ${OpenCV_LIBS}
                             ${Sophus_LIBRARIES} ${Pangolin_LIBRARIES})