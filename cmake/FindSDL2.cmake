find_path(SDL2_INCLUDE_DIRS
    NAMES SDL.h
    PATH_SUFFIXES SDL2
    )

find_library(SDL2_LIBRARIES
    NAMES SDL2
    PATH_SUFFIXES SDL2
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL2
    DEFAULT_MSG SDL2_LIBRARIES SDL2_INCLUDE_DIRS)
