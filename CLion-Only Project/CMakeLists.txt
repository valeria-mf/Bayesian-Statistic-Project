cmake_minimum_required(VERSION 3.23)
project(CLion_Only_Project)

set(CMAKE_CXX_STANDARD 14)

include_directories(C:/Users/lmaci/CLionProjects/Toolbox/eigen-3.4.0)


add_executable(CLion_Only_Project
        auxiliar_functions.cpp
        auxiliar_functions.h
        main.cpp
        truncated_normal.cpp
        truncated_normal.h)
