#configure variables
set (SAMPLE_NAME "app")

#configure directories
set (SAMPLE_PATH "${PROJECT_SOURCE_DIR}/sample")
set (SAMPLE_SRC_PATH  "${SAMPLE_PATH}/src" )
set (SAMPLE_INCLUDE_PATH "${SAMPLE_PATH}/include")

find_package(OpenCV REQUIRED)

set(Boost_USE_STATIC_LIBS   ON)
FIND_PACKAGE(Boost 1.58 COMPONENTS filesystem REQUIRED)

#set includes
include_directories (
    ${SAMPLE_INCLUDE_PATH}
    ${MTCNN_LIB_INCLUDE_PATH}
)

#set sources
file (GLOB SAMPLE_SOURCE_FILES "${SAMPLE_SRC_PATH}/*.cc")
file (GLOB SAMPLE_HEADER_FILES "${SAMPLE_INCLUDE_PATH}/*.h")

#set target executable
add_executable (${SAMPLE_NAME} ${SAMPLE_SOURCE_FILES} ${SAMPLE_HEADER_FILES})

if (WIN32)
set(LINK_OPTIONS shlwapi.lib)
endif()

if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
set(LINK_OPTIONS -pthread -ldl -static-libgcc -static-libstdc++)
endif()

if (Darwin)
set(LINK_OPTIONS -pthread)
endif()

#add the library
target_link_libraries (
    ${SAMPLE_NAME}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${MTCNN_LIB_NAME}
    ${LINK_OPTIONS})

#test
enable_testing ()
