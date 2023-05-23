cmake_minimum_required(VERSION 3.15)
project(ParaHW)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/obj)
set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES "src/*.cpp")
add_executable(bin ${SOURCES})

find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(${INCLUDE_PATH}
        include
        ${BLAS_INCLUDE_DIRS})
    # target_link_libraries(bin ${BLAS_LIBRARIES})
endif(BLAS_FOUND)
message(${BLAS_LIBRARIES})

find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message("LAPACK found.")
endif(LAPACK_FOUND)
message(${LAPACK_LIBRARIES})

option(${PROJECT_NAME}_MPI "use MPI" OFF)
 
if (DEFINED ${PROJECT_NAME}_MPI)
    message(STATUS "Defined MPI")
else()
    message(STATUS "Not Defined MPI")
endif()
 
if (${PROJECT_NAME}_MPI)
    message(STATUS "Using MPI")
    set(CMAKE_CXX_COMPILER mpicxx)
    set(CMAKE_C_COMPILER mpicc)
    add_definitions(-D__MPI)
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        message("MPI found.")
    endif(MPI_FOUND)
    target_link_libraries(bin ${MPI_CXX_LIBRARIES})
else()
    message(STATUS "Not Using MPI")
    add_definitions(-U__MPI)
endif()

target_link_libraries(bin cblas lapacke ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} gfortran)

add_subdirectory(src)
# add_subdirectory(third_party)
# message(${BLAS_INCLUDE_DIRS})


