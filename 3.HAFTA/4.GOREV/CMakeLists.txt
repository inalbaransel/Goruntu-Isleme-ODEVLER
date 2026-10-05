cmake_minimum_required(VERSION 3.16)
project(OrtalamaFiltre_Gorev4 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ortalama_filtre ortalama_filtre.cpp)
target_link_libraries(ortalama_filtre ${OpenCV_LIBS})
