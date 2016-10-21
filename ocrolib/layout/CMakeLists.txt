cmake_minimum_required(VERSION 2.8)
project(RLSA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

aux_source_directory(. DIR_LIB_SRCS)
add_library(RLSA SHARED ${DIR_LIB_SRCS})
