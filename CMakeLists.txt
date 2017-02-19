cmake_minimum_required(VERSION 2.8)

project(Clam3 C CXX)

set(SOURCEDIR ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${SOURCEDIR}/cmake)

set(CMAKE_BUILD_TYPE Release)

set(nowarn
        "-Wno-c++98-compat-pedantic"
        "-Wno-padded"
        "-Wno-deprecated"
        "-Wno-weak-vtables"
        "-Wno-old-style-cast"
        "-Wno-shadow-field-in-constructor"
        "-Wno-float-equal"
        )

if (MSVC)
    add_definitions(-DMSVC)
    add_compile_options("-EHsc")
else ()
    add_compile_options("-Weverything")
    foreach (item ${nowarn})
        add_compile_options(${item})
    endforeach ()
endif ()
set(NVCC_ARGS "--use_fast_math" "-lineinfo" "--gpu-architecture" "sm_52" "--resource-usage")

file(GLOB cppfiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.cpp")
file(GLOB hfiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.h")
set(sources ${cppfiles} ${hfiles})

set(CMAKE_LIBRARY_PATH
        "/home/kuhl/public-vrlab/vrpn/build"
        "/home/echauck/SDL2/lib"
        "${CMAKE_SOURCE_DIR}/../SDL2/lib"
        "${CMAKE_SOURCE_DIR}/../vrpn/build"
        "/opt/cuda/lib"
        ${CMAKE_LIBRARY_PATH})
set(CMAKE_INCLUDE_PATH
        "/home/echauck/SDL2/include"
        "/home/kuhl/public-vrlab/vrpn"
        "${CMAKE_SOURCE_DIR}/../SDL2/include"
        "${CMAKE_SOURCE_DIR}/../vrpn"
        "/opt/cuda/include"
        ${CMAKE_INCLUDE_PATH})

macro(my_package pacname)
    find_package(${pacname} REQUIRED)
    string(TOUPPER ${pacname} pacname_upper)
    include_directories(SYSTEM ${${pacname_upper}_INCLUDE_DIRS})
endmacro()

macro(my_package_noreq pacname)
    find_package(${pacname})
    string(TOUPPER ${pacname} pacname_upper)
    if (${pacname_upper}_FOUND)
        include_directories(SYSTEM ${${pacname_upper}_INCLUDE_DIRS})
        add_definitions(-DHAVE_${pacname_upper})
    else ()
        message(WARNING "${pacname} not found!")
    endif ()
endmacro()

my_package(CUDA)
my_package(SDL2)
#my_package(SDL2_net)
my_package(SDL2_ttf)
my_package_noreq(VRPN)

add_executable(embed util/embed.c)

set(built_shfiles)
file(GLOB shfiles RELATIVE "${SOURCEDIR}/util" "${SOURCEDIR}/util/*.sh")
foreach (shfile ${shfiles})
    set(infile "${SOURCEDIR}/util/${shfile}")
    set(outfile "${CMAKE_BINARY_DIR}/${shfile}")
    add_custom_command(
            OUTPUT "${outfile}"
            DEPENDS "${infile}"
            COMMAND ${CMAKE_COMMAND} -E copy ${infile} ${outfile})
    list(APPEND built_shfiles ${outfile})
endforeach ()

set(genfiles)
include_directories("${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
file(GLOB cufiles RELATIVE ${SOURCEDIR} "${SOURCEDIR}/*.cu")
foreach (filename ${cufiles})
    set(infile "${SOURCEDIR}/${filename}")
    set(fatbinfile "${CMAKE_BINARY_DIR}/generated/${filename}.fatbin")
    set(outfile "${CMAKE_BINARY_DIR}/generated/${filename}.c")
    get_filename_component(varname ${filename} NAME_WE)
    add_custom_command(
            OUTPUT "${fatbinfile}"
            DEPENDS ${infile}
            COMMAND nvcc ${NVCC_ARGS} "-fatbin" "-o" "${fatbinfile}" "${infile}")
    add_custom_command(
            OUTPUT "${outfile}"
            DEPENDS ${fatbinfile}
            COMMAND embed "${fatbinfile}" "${outfile}" "${varname}")
    list(APPEND genfiles ${outfile})
endforeach ()

add_executable(clam3 ${sources} ${genfiles} ${built_shfiles})

set_property(TARGET clam3 PROPERTY CXX_STANDARD 11)
set_property(TARGET clam3 PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET clam3 PROPERTY C_STANDARD 99)
set_property(TARGET clam3 PROPERTY C_STANDARD_REQUIRED ON)

target_link_libraries(clam3
        ${SDL2_LIBRARIES}
        ${SDL2_NET_LIBRARIES}
        ${SDL2_TTF_LIBRARIES})

if (MSVC)
    target_link_libraries(clam3 "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib")
else ()
    target_link_libraries(clam3 cuda)
endif ()

if (VRPN_FOUND)
    target_link_libraries(clam3
            pthread
            ${VRPN_LIBRARIES})
endif ()
