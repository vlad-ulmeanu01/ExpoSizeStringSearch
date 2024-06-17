cmake_minimum_required ( VERSION 3.4.3 )
project ( E3Saggrocl CXX )

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

if ( APPLE )
    set ( CMAKE_MACOSX_RPATH OFF )
endif ( APPLE )

include ( FindPkgConfig )
pkg_search_module ( SNORT3 REQUIRED snort>=3 )
find_package(OpenCL REQUIRED)

add_library (
    E3Saggrocl MODULE
    E3Saggrocl.cc
    E3Saggrocl_api.cc
    E3Saggrocl_utils.cc
    E3Saggrocl_utils.h
)

if ( APPLE )
    set_target_properties (
        E3Saggrocl
        PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
    )
endif ( APPLE )

set_target_properties (
    E3Saggrocl
    PROPERTIES
        PREFIX ""
)

target_include_directories (
    E3Saggrocl PUBLIC
    ${SNORT3_INCLUDE_DIRS}
)

target_link_libraries(E3Saggrocl OpenCL::OpenCL)

install (
    TARGETS E3Saggrocl
    LIBRARY
        DESTINATION "${SEARCH_ENGINE_INSTALL_PATH}"
)
