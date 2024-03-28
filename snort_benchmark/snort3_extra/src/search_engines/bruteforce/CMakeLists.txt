cmake_minimum_required ( VERSION 3.4.3 )
project ( bruteforce CXX )

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

if ( APPLE )
    set ( CMAKE_MACOSX_RPATH OFF )
endif ( APPLE )

include ( FindPkgConfig )
pkg_search_module ( SNORT3 REQUIRED snort>=3 )

add_library (
    bruteforce MODULE
    bruteforce.cc
    bruteforce_api.cc
)

if ( APPLE )
    set_target_properties (
        bruteforce
        PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
    )
endif ( APPLE )

set_target_properties (
    bruteforce
    PROPERTIES
        PREFIX ""
)

target_include_directories (
    bruteforce PUBLIC
    ${SNORT3_INCLUDE_DIRS}
)

install (
    TARGETS bruteforce
    LIBRARY
        DESTINATION "${SEARCH_ENGINE_INSTALL_PATH}"
)
