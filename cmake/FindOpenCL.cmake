find_path(OPENCL_INCLUDE_DIRS
    NAMES CL/cl.h
    )

find_library(OPENCL_LIBRARIES
    NAMES OpenCL
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OPENCL
    DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS)
