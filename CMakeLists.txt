cmake_minimum_required(VERSION 2.6)

project(Clam3 C CXX)

set(SOURCEDIR ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${SOURCEDIR}/cmake)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -Wall -Wextra -Wno-deprecated-declarations")
set(CUDA_NVCC_FLAGS "-O3;--compiler-options;-Wall;--use_fast_math;-lineinfo")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

file(GLOB cppfiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.cpp")
file(GLOB hfiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.h")
set(sources ${cppfiles} ${hfiles})

set(CMAKE_LIBRARY_PATH
        "/home/echauck/SDL2/lib"
        "/home/echauck/SDL2_net/lib"
        "/home/echauck/SDL2_ttf/lib"
        "/home/kuhl/public-vrlab/vrpn/build"
        "${CMAKE_SOURCE_DIR}/../vrpn/build"
        "/opt/cuda/lib"
        ${CMAKE_LIBRARY_PATH})
set(CMAKE_INCLUDE_PATH
        "/home/echauck/SDL2/include"
        "/home/echauck/SDL2_net/include"
        "/home/echauck/SDL2_ttf/include"
        "/opt/intel/opencl-sdk/include"
        "/home/kuhl/public-vrlab/vrpn"
        "${CMAKE_SOURCE_DIR}/../vrpn"
        "/opt/cuda/include"
        ${CMAKE_INCLUDE_PATH})

macro(my_package pacname)
    find_package(${pacname} REQUIRED)
    string(TOUPPER ${pacname} pacname_upper)
    include_directories(${${pacname_upper}_INCLUDE_DIRS})
endmacro()

macro(my_package_noreq pacname)
    find_package(${pacname})
    string(TOUPPER ${pacname} pacname_upper)
    if (${pacname_upper}_FOUND)
        include_directories(${${pacname_upper}_INCLUDE_DIRS})
    endif ()
endmacro()

my_package(OpenGL)
my_package(CUDA)
my_package(SDL2)
my_package(SDL2_net)
my_package(SDL2_ttf)
my_package_noreq(VRPN)

if (VRPN_FOUND)
    add_definitions(-DHAVE_VRPN)
endif()

add_executable(embed util/embed.c)

file(COPY util/ivs.sh DESTINATION .)

set(genfiles)
include_directories("${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
file(GLOB clfiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.cu")
foreach (filename ${clfiles})
    cuda_compile_ptx(ptxfile "${filename}")
    #set(ptxfile "${CMAKE_BINARY_DIR}/generated/${filename}.ptx")
    #add_custom_command(
    #        OUTPUT "${ptxfile}"
    #        DEPENDS ${filename}
    #        WORKING_DIRECTORY ${SOURCEDIR}
    #        COMMAND nvcc --ptx ${NVCC_FLAGS} "${filename}" -o "${ptxfile}")
    set(outfile "${CMAKE_BINARY_DIR}/generated/${filename}.c")
    get_filename_component(varname ${filename} NAME_WE)
    add_custom_command(
            OUTPUT "${outfile}"
            DEPENDS ${ptxfile}
            COMMAND embed "${ptxfile}" "${outfile}" "${varname}")
    list(APPEND genfiles ${outfile})
endforeach ()

add_executable(clam3 ${sources} ${genfiles})

target_link_libraries(clam3
        ${OPENGL_LIBRARIES}
        cuda
        ${SDL2_LIBRARIES}
        ${SDL2_NET_LIBRARIES}
        ${SDL2_TTF_LIBRARIES})

if (VRPN_FOUND)
    target_link_libraries(clam3
            pthread
            ${VRPN_LIBRARIES})
endif ()