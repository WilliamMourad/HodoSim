cmake_minimum_required(VERSION 3.16...3.27)

project(RootApp)

find_package(ROOT REQUIRED)

add_executable(RootApp WIN32 main.cc)
#add_executable(RootApp main.cc)

include(${ROOT_USE_FILE})

target_link_libraries(RootApp ${ROOT_LIBRARIES})

