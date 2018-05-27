#configure variables
set (LIB_NAME "mtcnn")

#configure directories
set (LIB_PATH "${PROJECT_SOURCE_DIR}/lib")
set (LIB_SRC_PATH  "${LIB_PATH}/src" )
set (LIB_INCLUDE_PATH  "${LIB_PATH}/include")

find_package(OpenCV REQUIRED)

#set includes
include_directories (
    ${LIB_INCLUDE_PATH}
    ${OpenCV_INCLUDE_DIRS}    
    ${DLIB_INCLUDE_DIR}
)

#set sources
file (GLOB LIB_HEADER_FILES "${LIB_INCLUDE_PATH}/**/*.h" "${LIB_INCLUDE_PATH}/**/**/*.h")
file (GLOB LIB_SOURCE_FILES "${LIB_SRC_PATH}/*.cc" "${LIB_SRC_PATH}/**/*.cc")

#set library
add_library (${LIB_NAME} STATIC ${LIB_SOURCE_FILES} ${LIB_HEADER_FILES})

#export vars
set (LIB_INCLUDE_PATH  ${LIB_INCLUDE_PATH} PARENT_SCOPE)
set (LIB_NAME ${LIB_NAME} PARENT_SCOPE)

#test
enable_testing ()
