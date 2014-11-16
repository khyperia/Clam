find_path(ASSIMP_INCLUDE_DIRS
    NAMES cimport.h
    PATH_SUFFIXES assimp
    )

find_library(ASSIMP_LIBRARIES
    NAMES assimp
    PATH_SUFFIXES assimp
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ASSIMP
    DEFAULT_MSG ASSIMP_LIBRARIES ASSIMP_INCLUDE_DIRS)
