cmake_minimum_required(VERSION 3.16)
project(CLAHE_Gorev3 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV kütüphanesini bul
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(clahe clahe.cpp)
target_link_libraries(clahe ${OpenCV_LIBS})
