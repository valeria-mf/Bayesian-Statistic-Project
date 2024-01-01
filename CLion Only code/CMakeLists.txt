cmake_minimum_required(VERSION 3.23)
include_directories(C:/Users/lmaci/CLionProjects/Toolbox/eigen-3.4.0)

project(bayesiana)

set(CMAKE_CXX_STANDARD 14)
add_executable(bayesiana main.cpp auxiliar_functions.cpp auxiliar_functions.h GibbsSampler_betabernoulli.cpp GibbsSampler_betabernoulli.h)
