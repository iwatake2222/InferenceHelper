if(DEFINED  ANDROID_ABI)
    message(FATAL_ERROR "[ARMNN] todo")
elseif(${BUILD_SYSTEM} STREQUAL "x64_linux")
    set(ARMNN_LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../armnn_prebuilt/ubuntu/lib/)
    set(ARMNN_PROTOBUF_LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../armnn_prebuilt/protobuf_lib/ubuntu/)
    set(ARMNN_INC_PATH ${CMAKE_CURRENT_LIST_DIR}/../armnn_prebuilt/ubuntu/include/)
elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
    set(ARMNN_LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../armnn_prebuilt/aarch64/lib/)
    set(ARMNN_PROTOBUF_LIB_PATH ${CMAKE_CURRENT_LIST_DIR}/../armnn_prebuilt/protobuf_lib/aarch64/)
    set(ARMNN_INC_PATH ${CMAKE_CURRENT_LIST_DIR}/../armnn_prebuilt/aarch64/include/)
else()
    message(FATAL_ERROR "[ARMNN] unsupported platform")
endif()

function(ADD_SO_LIB SO_LIB_NAME SO_FILE_PATH)
    add_library(${SO_LIB_NAME} SHARED IMPORTED GLOBAL)
    set_target_properties(
        ${SO_LIB_NAME}
        PROPERTIES IMPORTED_LOCATION
        ${SO_FILE_PATH}
    )
    set(ARMNN_LIB ${ARMNN_LIB} ${SO_LIB_NAME} PARENT_SCOPE)
endfunction()

set(ARMNN_LIB )
ADD_SO_LIB("ARMNN" ${ARMNN_LIB_PATH}/libarmnn.so)
ADD_SO_LIB("ARMNN_ONNX_PARSER" ${ARMNN_LIB_PATH}/libarmnnOnnxParser.so)
ADD_SO_LIB("ARMNN_TFLITE_PARSER" ${ARMNN_LIB_PATH}/libarmnnTfLiteParser.so)
ADD_SO_LIB("PROTOBUF_LITE" ${ARMNN_PROTOBUF_LIB_PATH}/libprotobuf-lite.so)
ADD_SO_LIB("PROTOBUF" ${ARMNN_PROTOBUF_LIB_PATH}/libprotobuf.so)
ADD_SO_LIB("PROTOC" ${ARMNN_PROTOBUF_LIB_PATH}/libprotoc.so)
set(ARMNN_LIB ${ARMNN_LIB} pthread)
set(ARMNN_INC ${ARMNN_INC_PATH})

# workaround: libprotobuf.so.23 is not found
file(COPY ${ARMNN_PROTOBUF_LIB_PATH}/libprotobuf.so.23 DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${ARMNN_PROTOBUF_LIB_PATH}/libprotobuf.so.23.0.0 DESTINATION ${CMAKE_BINARY_DIR})

### dump
# set(Armnn_DIR ${CMAKE_CURRENT_LIST_DIR}/install/lib/cmake/armnn)
# find_package(Armnn REQUIRED)
# target_link_libraries(${LibraryName} ${ARMNN_LIBRARIES})
# target_link_libraries(${LibraryName} pthread)
