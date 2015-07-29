cmake_minimum_required(VERSION 2.6)

project(Clam3 C CXX)

set(SOURCEDIR ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${SOURCEDIR}/cmake)

set(CMAKE_BUILD_TYPE Release)

if (${CMAKE_BUILD_TYPE} STREQUAL Debug)
    add_definitions(-DDEBUG)
else()
    add_definitions(-DRELEASE)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -Wall -Wextra -Wno-deprecated-declarations")

file(GLOB cppfiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.cpp")
file(GLOB hfiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.h")
set(sources ${cppfiles} ${hfiles})

set(CMAKE_LIBRARY_PATH
    "/home/echauck/SDL2/lib"
    "/home/echauck/SDL2_net/lib"
    ${CMAKE_LIBRARY_PATH})
set(CMAKE_INCLUDE_PATH
    "/home/echauck/SDL2/include"
    "/home/echauck/SDL2_net/include"
    "/opt/intel/opencl-sdk/include"
    ${CMAKE_INCLUDE_PATH})

add_definitions(-D__CL_ENABLE_EXCEPTIONS)

macro(my_package pacname)
    find_package(${pacname})
    string(TOUPPER ${pacname} pacname_upper)
    if (${pacname_upper}_FOUND)
        include_directories(${${pacname_upper}_INCLUDE_DIRS})
    endif()
endmacro()

my_package(OpenGL)
my_package(OpenCL)
my_package(SDL2)
my_package(SDL2_net)

add_executable(embed util/embed.c)

set(genfiles)
include_directories("${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
file(GLOB clfiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.cl")
foreach(filename ${clfiles})
    set(outfile "${CMAKE_BINARY_DIR}/generated/${filename}.c")
    get_filename_component(varname ${filename} NAME_WE)
    add_custom_command(
        OUTPUT "${outfile}"
        DEPENDS ${filename}
        WORKING_DIRECTORY ${SOURCEDIR}
        COMMAND embed "${filename}" "${outfile}" "${varname}")
    list(APPEND genfiles ${outfile})
endforeach()

add_executable(clam3 ${sources} ${genfiles})

target_link_libraries(clam3
    ${OPENGL_LIBRARIES}
    ${OPENCL_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${SDL2_NET_LIBRARIES})
