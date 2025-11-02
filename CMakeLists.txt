cmake_minimum_required(VERSION 3.16...3.27)

project(HodoSim)

# Add PlotPredict subdirectory
add_subdirectory(PlotPredict)
# Comment this next line if you got the code from GitHub
# the Analyzer subdirectory is just for internal use.
# add_subdirectory(Analyzer)

find_package(Geant4 REQUIRED ui_all vis_all)
find_package(ryml REQUIRED) # I added rapidyaml dependency to parse YAML configuration files


file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)

add_executable(HodoSim main.cc ${sources} ${headers})
target_include_directories(HodoSim PRIVATE include)
target_link_libraries(HodoSim PRIVATE ${Geant4_LIBRARIES})
target_link_libraries(HodoSim PRIVATE ryml::ryml)

set(SCRIPTS
	macros/init.mac
	macros/vis.mac
	macros/batch.mac
	macros/paint_geometry.mac
	macros/build_custom_gui.mac
	macros/_plot.mac
	macros/extra/calibration.mac

)

foreach(_script ${SCRIPTS})
	configure_file(
		${PROJECT_SOURCE_DIR}/${_script}
		${PROJECT_BINARY_DIR}/${_script}
		COPYONLY
	)
endforeach()

configure_file(
	${PROJECT_SOURCE_DIR}/config.yaml
	${PROJECT_BINARY_DIR}/config.yaml
	COPYONLY
)