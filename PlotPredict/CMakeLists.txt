# CMake configuration for PlotPredict application

cmake_minimum_required(VERSION 3.16...3.27)

project(PlotPredict)

find_package(ROOT REQUIRED)

#add_executable(PlotPredict WIN32 main.cc)
add_executable(PlotPredict main.cc)

include(${ROOT_USE_FILE})
target_include_directories(PlotPredict PRIVATE ${ONNXRUNTIME_DIR}/include)

target_link_libraries(PlotPredict ${ROOT_LIBRARIES} ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)

configure_file(
		${ONNXRUNTIME_DIR}/lib/onnxruntime.dll
		${PROJECT_BINARY_DIR}/onnxruntime.dll
		COPYONLY
)