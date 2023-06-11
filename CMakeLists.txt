cmake_minimum_required(VERSION 3.15)
project(ParaHW)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/obj)
set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "src/*.cpp")
add_executable(bin ${SOURCES})
 
if (${PROJECT_NAME}_MPI)
    message(STATUS "Using MPI")
    set(CMAKE_CXX_COMPILER mpicxx)
    set(CMAKE_C_COMPILER mpicc)
    add_definitions(-D__MPI)
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        message("MPI found.")
    endif(MPI_FOUND)
else()
    message(STATUS "Not Using MPI")
    add_definitions(-U__MPI)
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message("OpenMP found.")
    include_directories(${INCLUDE_PATH}
        include
        ${OpenMP_CXX_INCLUDE_DIRS})
    # target_link_libraries(bin ${BLAS_LIBRARIES})
endif(OpenMP_FOUND)
message(${OpenMP_CXX_LIBRARIES})

set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicc)

target_link_libraries(bin ${MPI_CXX_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES} gfortran)
set(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
message(${OpenMP_CXX_FLAGS})

add_subdirectory(src)
# add_subdirectory(third_party)
# message(${BLAS_INCLUDE_DIRS})


