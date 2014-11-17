find_path(LUA_INCLUDE_DIRS
    NAMES lua.h
    )

find_library(LUA_LIBRARIES
    NAMES lua
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(LUA
    DEFAULT_MSG LUA_LIBRARIES LUA_INCLUDE_DIRS)
