cmake_minimum_required(VERSION 3.15)
project(ParaHW)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/obj)
set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "src/*.cpp")
add_executable(bin ${SOURCES})
 

message(STATUS "Using MPI")
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicc)
add_definitions(-D__MPI)
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message("MPI found.")
endif(MPI_FOUND)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message("OpenMP found.")
    include_directories(${INCLUDE_PATH}
        include
        ${OpenMP_CXX_INCLUDE_DIRS})
    # target_link_libraries(bin ${BLAS_LIBRARIES})
endif(OpenMP_FOUND)
message(${OpenMP_CXX_LIBRARIES})

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("Blas found.")
    include_directories(${INCLUDE_PATH}
        include
        ${BLAS_INCLUDE_DIRS})
endif(BLAS_FOUND)

find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message("LAPACK found.")
endif(LAPACK_FOUND)

find_package(LAPACKE REQUIRED)
if(LAPACKE_FOUND)
    message("OpenMP found.")
    include_directories(${INCLUDE_PATH}
        include
        ${LAPACKE_INCLUDE_DIRS})
endif(LAPACKE_FOUND)

set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicc)

target_link_libraries(bin ${MPI_CXX_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES} gfortran)
set(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
message(${OpenMP_CXX_FLAGS})

add_subdirectory(src)
# add_subdirectory(third_party)
# message(${BLAS_INCLUDE_DIRS})


