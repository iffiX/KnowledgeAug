cmake_minimum_required(VERSION 3.12)
project(matcher)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
message("Build in ${CMAKE_BUILD_TYPE} mode")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
    message( SEND_ERROR "Compiler version too low, requires g++ > 8.0 to compile" )
endif()
################################################################
# Configuration for depended libraries
################################################################
set(CMAKE_LIBRARY_PATH "/home/mlw0504/miniconda3/lib")
set(CMAKE_INCLUDE_PATH "/home/mlw0504/miniconda3/include")
find_package(HDF5 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)
#set(XTENSOR_USE_XSIMD 1)
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -fPIC)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fPIC)
add_subdirectory(backward-cpp)
find_package(OpenMP)
if (OPENMP_FOUND)
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
################################################################
# Main builds
################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/xtl/include)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    pybind11_add_module(matcher SHARED init.cpp matcher.cpp concept_net.cpp ${BACKWARD_ENABLE})
    target_include_directories(matcher PUBLIC
            ${Python3_INCLUDE_DIRS}
            ${PYBIND11_INCLUDE_DIR}
            ${HDF5_INCLUDE_DIRS})
    add_backward(matcher)
    target_link_libraries(matcher PRIVATE ${Python3_LIBRARIES} ${HDF5_LIBRARIES})
else()
    pybind11_add_module(matcher SHARED init.cpp matcher.cpp concept_net.cpp)
    target_include_directories(matcher PUBLIC
            ${Python3_INCLUDE_DIRS}
            ${PYBIND11_INCLUDE_DIR}
            ${HDF5_INCLUDE_DIRS})
    target_link_libraries(matcher PRIVATE ${Python3_LIBRARIES} ${HDF5_LIBRARIES})
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
