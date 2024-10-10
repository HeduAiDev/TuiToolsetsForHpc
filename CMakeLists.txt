cmake_minimum_required (VERSION 3.18)
project(TuiToolsetsForHpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# use MT runtime for win
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
message(STATUS "CMAKE_MSVC_RUNTIME_LIBRARY: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
# ftxui
add_subdirectory(3rd/ftxui EXCLUDE_FROM_ALL)
# pybind11
add_subdirectory(3rd/pybind11 EXCLUDE_FROM_ALL)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
# static lib tui_toolsets
file(GLOB src_components CONFIGURE_DEPENDS "csrc/Component/*.cpp")
file(GLOB src_runables CONFIGURE_DEPENDS "csrc/runable/*.cpp")
add_library(tui_toolsets STATIC  ${src_components} ${src_runables})
target_link_libraries(tui_toolsets PRIVATE ftxui::screen
                                    PRIVATE ftxui::dom
                                    PRIVATE ftxui::component)
target_include_directories(tui_toolsets PUBLIC "csrc/include")
include(tools/merge_static_libs.cmake)
# protable static lib (merged all in one) tui_toolsets_static 
merge_static_libs(tui_toolsets_static tui_toolsets ftxui::screen ftxui::dom ftxui::component)
set_target_properties(tui_toolsets_static PROPERTIES OUTPUT_NAME tui_toolsets)
# python c extension tui_toolsets_runables
add_library(_C MODULE csrc/pybind.cpp)
target_include_directories(_C PRIVATE "csrc/include")
target_link_libraries(_C PRIVATE tui_toolsets
                                    PRIVATE ftxui::screen
                                    PRIVATE ftxui::dom
                                    PRIVATE ftxui::component
                                    PRIVATE pybind11::module)
set_target_properties(_C PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})
# install
install(TARGETS _C DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src/tui_toolsets COMPONENT PyExtension)

install(TARGETS tui_toolsets_static DESTINATION lib COMPONENT CPPInterface)
install(DIRECTORY csrc/include/ DESTINATION include COMPONENT CPPInterface
        PATTERN "color_info_*" EXCLUDE)
