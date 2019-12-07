cmake_minimum_required(VERSION 3.8)

set (PROJ_NAME SpatialSubspaceRotation)
project(${PROJ_NAME})

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package(OpenCV REQUIRED)
message("Opencv Version: ${OPENCV_VERSION}" )

set(folder_source src/main.cpp src/SpatialSubspaceRotation.cpp)

set(folder_header src/SpatialSubspaceRotation.h)

SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_header})

set(INC_DIRS
    ${PROJECT_SOURCE_DIR}/src
)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
file(GLOB DSPFILTERS_HEADER ${CMAKE_SOURCE_DIR}/iir/iir/*.h*)
file(GLOB DSPFILTERS_SOURCE ${CMAKE_SOURCE_DIR}/iir/iir/*.c*)
include_directories(${CMAKE_SOURCE_DIR}/iir)
include_directories(${CMAKE_SOURCE_DIR}/iir/iir)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -std=c++11 -fopenmp")
include_directories(${INC_DIRS})

add_executable(${PROJ_NAME} ${folder_source} ${folder_header} ${DSPFILTERS_HEADER} ${DSPFILTERS_SOURCE} )
target_link_libraries(${PROJ_NAME} ${OpenCV_LIBS})
