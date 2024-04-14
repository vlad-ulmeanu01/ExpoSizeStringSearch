cmake_minimum_required ( VERSION 3.4.3 )
project ( E3Saggro CXX )

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

if ( APPLE )
    set ( CMAKE_MACOSX_RPATH OFF )
endif ( APPLE )

include ( FindPkgConfig )
pkg_search_module ( SNORT3 REQUIRED snort>=3 )

add_library (
    E3Saggro MODULE
    E3Saggro.cc
    E3Saggro_api.cc
    E3Saggro_utils.cc
    E3Saggro_utils.h
)

if ( APPLE )
    set_target_properties (
        E3Saggro
        PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
    )
endif ( APPLE )

set_target_properties (
    E3Saggro
    PROPERTIES
        PREFIX ""
)

target_include_directories (
    E3Saggro PUBLIC
    ${SNORT3_INCLUDE_DIRS}
)

install (
    TARGETS E3Saggro
    LIBRARY
        DESTINATION "${SEARCH_ENGINE_INSTALL_PATH}"
)
