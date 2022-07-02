cmake_minimum_required(VERSION 3.12)
project(matcher)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
message("Build in ${CMAKE_BUILD_TYPE} mode")
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
    message( SEND_ERROR "Compiler version too low, requires g++ > 8.0 to compile" )
endif()
################################################################
# Configuration for depended libraries
################################################################
find_package(HDF5 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)
#set(XTENSOR_USE_XSIMD 1)
add_subdirectory(backward-cpp)
################################################################
# Main builds
################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/xtl/include)
pybind11_add_module(matcher SHARED init.cpp matcher.cpp concept_net.cpp)
target_include_directories(matcher PUBLIC
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIRS})
target_link_libraries(matcher PRIVATE ${Python3_LIBRARIES} ${HDF5_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_backward(matcher)
endif()
install(TARGETS matcher LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

add_library(matcher_test matcher.cpp concept_net.cpp)
target_include_directories(matcher_test PUBLIC
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIRS})
target_link_libraries(matcher_test PUBLIC ${Python3_LIBRARIES} ${HDF5_LIBRARIES})
add_executable(test main.cpp)
target_link_libraries(test matcher_test)
