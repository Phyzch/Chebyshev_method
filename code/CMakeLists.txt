cmake_minimum_required(VERSION 3.10.2)
project(code)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

SET(CMAKE_CXX_COMPILER mpicxx)
SET(CMAKE_C_COMPILER  mpicc)


include_directories(.)

add_executable(code
        Evolve_Chebyshev_method_real_time.cpp
        Evolve_wave_func_imag_time_Chebyshev_polynomial.cpp
        molecule.h
        prepare_evolution.cpp
        util.h
        update_wave_func.cpp
        Quantum_Evolution.cpp)

target_link_libraries(${PROJECT_NAME} stdc++fs mpi)
