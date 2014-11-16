cmake_minimum_required(VERSION 2.8)
project(Clam2 C CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -Wall -Wextra")

# set(CMAKE_LIBRARY_PATH "fzoo" ${CMAKE_LIBRARY_PATH})
# set(CMAKE_INCLUDE_PATH "fzoo" ${CMAKE_INCLUDE_PATH})

add_executable(clam2_master master.c masterSocket.c luaHelper.c socketHelper.c helper.c)
add_executable(clam2_slave slave.c slaveSocket.c glclContext.c openclHelper.c socketHelper.c helper.c)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

find_package(SDL2 REQUIRED SDL2_BUILDING_LIBRARY)
include_directories(${SDL2_INCLUDE_DIRS})

find_package(SDL2_net REQUIRED)
include_directories(${SDL2_NET_INCLUDE_DIRS})

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

find_package(Lua REQUIRED)
include_directories(${LUA_INCLUDE_DIRS})

target_link_libraries(clam2_master
    ${OPENGL_LIBRARIES}
    ${PNG_LIBRARIES}
    ${LUA_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${SDL2_NET_LIBRARIES})

# OpenCL gl sdl2 sdl2_net
target_link_libraries(clam2_slave
    ${OPENCL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${SDL2_NET_LIBRARIES})

file(GLOB files "script/*.c" "script/*.cpp")
foreach(file ${files})
    get_filename_component(libname ${file} NAME_WE)
    add_library(${libname} SHARED ${file})
    set_target_properties(${libname} PROPERTIES LIBRARY_OUTPUT_DIRECTORY script)
    target_link_libraries(${libname} ${LUA_LIBRARIES})
endforeach()

file(COPY script DESTINATION .)
