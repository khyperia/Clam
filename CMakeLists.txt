cmake_minimum_required(VERSION 2.6)
project(Clam2 C CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++98 -Wall -Wextra")

set(CMAKE_LIBRARY_PATH
    "/home/echauck/lua/src"
    "/home/echauck/SDL2/lib"
    "/home/echauck/SDL2_net/lib"
    "/home/kuhl/public-vrlab/vrpn/build"
    ${CMAKE_LIBRARY_PATH})
set(CMAKE_INCLUDE_PATH
    "/home/echauck/lua/src"
    "/home/echauck/SDL2/include"
    "/home/echauck/SDL2_net/include"
    "/home/kuhl/public-vrlab/vrpn"
    "/export/apps/cuda-5.0/include"
    ${CMAKE_INCLUDE_PATH})

add_executable(clam2_master master.c masterSocket.c luaHelper.c socketHelper.c helper.c)
add_executable(clam2_slave slave.c slaveSocket.c glclContext.c openclHelper.c socketHelper.c helper.c)

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
my_package(PNG)
my_package(Lua)
my_package(assimp)
my_package(VRPN)

if (LUA_FOUND)
target_link_libraries(clam2_master
    ${OPENGL_LIBRARIES}
    ${PNG_LIBRARIES}
    ${LUA_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${SDL2_NET_LIBRARIES})
endif()

target_link_libraries(clam2_slave
    ${OPENCL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${SDL2_NET_LIBRARIES})

file(GLOB files "script/*.c" "script/*.cpp")

if (NOT (ASSIMP_FOUND))
    list(REMOVE_ITEM files "${CMAKE_CURRENT_SOURCE_DIR}/script/maketris.c")
    message(STATUS "Assimp not found, not building maketris")
endif()

if (NOT (VRPN_FOUND))
    list(REMOVE_ITEM files "${CMAKE_CURRENT_SOURCE_DIR}/script/vrpn_help.cpp")
    message(STATUS "VRPN not found, not building vrpn_help")
endif()

if (LUA_FOUND)

set(pluginList)

foreach(file ${files})
    get_filename_component(libname ${file} NAME_WE)
    add_library(${libname} SHARED ${file})
    set_target_properties(${libname} PROPERTIES LIBRARY_OUTPUT_DIRECTORY script)
    target_link_libraries(${libname} ${LUA_LIBRARIES})
    set(pluginList ${pluginList} ${libname})
endforeach()

if (TARGET maketris)
    target_link_libraries(maketris ${ASSIMP_LIBRARIES})
endif()
if (TARGET vrpn_help)
    target_link_libraries(vrpn_help ${VRPN_LIBRARIES})
endif()

add_custom_target(plugins DEPENDS ${pluginList})

file(GLOB scriptFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "script/*")
foreach(scriptFile ${scriptFiles})
	configure_file(${scriptFile} ${scriptFile})
endforeach()
else()
    message(STATUS "LUA not found, not building scripts")
endif() # LUA_FOUND
