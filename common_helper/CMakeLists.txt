cmake_minimum_required(VERSION 3.0)

set(LibraryName "CommonHelper")

# Create library
set(SRC common_helper.h common_helper.cpp)
add_library(${LibraryName} ${SRC})
