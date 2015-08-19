find_path(SDL2_TTF_INCLUDE_DIRS
    NAMES SDL_ttf.h
    PATH_SUFFIXES SDL2
    )

find_library(SDL2_TTF_LIBRARIES
    NAMES SDL2_ttf
    PATH_SUFFIXES SDL2
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL2_ttf
    DEFAULT_MSG SDL2_TTF_LIBRARIES SDL2_TTF_INCLUDE_DIRS)
