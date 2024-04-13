cmake_minimum_required ( VERSION 3.4.3 )
project ( E3S CXX )

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

if ( APPLE )
    set ( CMAKE_MACOSX_RPATH OFF )
endif ( APPLE )

include ( FindPkgConfig )
pkg_search_module ( SNORT3 REQUIRED snort>=3 )

add_library (
    E3S MODULE
    E3S.cc
    E3S_api.cc
    E3S_utils.cc
    E3S_utils.h
)

if ( APPLE )
    set_target_properties (
        E3S
        PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
    )
endif ( APPLE )

set_target_properties (
    E3S
    PROPERTIES
        PREFIX ""
)

target_include_directories (
    E3S PUBLIC
    ${SNORT3_INCLUDE_DIRS}
)

install (
    TARGETS E3S
    LIBRARY
        DESTINATION "${SEARCH_ENGINE_INSTALL_PATH}"
)
