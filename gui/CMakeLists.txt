cmake_minimum_required(VERSION 3.10)
project(DroneControlGUI)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_executable(DroneControlGUI main.cpp)
target_link_libraries(DroneControlGUI Qt5::Widgets)
