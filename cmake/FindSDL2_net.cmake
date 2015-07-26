find_path(SDL2_NET_INCLUDE_DIRS
    NAMES SDL_net.h
    PATH_SUFFIXES SDL2
    )

find_library(SDL2_NET_LIBRARIES
    NAMES SDL2_net
    PATH_SUFFIXES SDL2
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL2_net
    DEFAULT_MSG SDL2_NET_LIBRARIES SDL2_NET_INCLUDE_DIRS)
