#configure variables
set (MTCNN_LIB_NAME "mtcnn")

#configure directories
set (MTCNN_LIB_PATH "${PROJECT_SOURCE_DIR}/lib")
set (MTCNN_LIB_SRC_PATH  "${MTCNN_LIB_PATH}/src" )
set (MTCNN_LIB_INCLUDE_PATH  "${MTCNN_LIB_PATH}/include")

find_package(OpenCV REQUIRED)

#set includes
include_directories (
    ${MTCNN_LIB_INCLUDE_PATH}
    ${OpenCV_INCLUDE_DIRS}   
)

#set sources
file (GLOB MTCNN_LIB_HEADER_FILES "${MTCNN_LIB_INCLUDE_PATH}/**/*.h" "${MTCNN_LIB_INCLUDE_PATH}/**/**/*.h")
file (GLOB MTCNN_LIB_SOURCE_FILES "${MTCNN_LIB_SRC_PATH}/*.cc" "${MTCNN_LIB_SRC_PATH}/**/*.cc")

#set library
add_library (${MTCNN_LIB_NAME} STATIC ${MTCNN_LIB_SOURCE_FILES} ${MTCNN_LIB_HEADER_FILES})

#export vars
set (MTCNN_LIB_INCLUDE_PATH  ${MTCNN_LIB_INCLUDE_PATH} PARENT_SCOPE)
set (MTCNN_LIB_NAME ${MTCNN_LIB_NAME} PARENT_SCOPE)

#test
enable_testing ()
