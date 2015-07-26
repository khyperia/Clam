find_path(OPENGL_INCLUDE_DIRS
    NAMES GL/gl.h
    )

find_library(OPENGL_LIBRARIES
    NAMES GL opengl32
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OPENGL
    DEFAULT_MSG OPENGL_LIBRARIES OPENGL_INCLUDE_DIRS)
