find_path(VRPN_INCLUDE_DIRS
    NAMES vrpn_Tracker.h
    )

find_library(VRPN_LIBRARIES
    NAMES vrpn
    )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(VRPN
    DEFAULT_MSG VRPN_LIBRARIES VRPN_INCLUDE_DIRS)
