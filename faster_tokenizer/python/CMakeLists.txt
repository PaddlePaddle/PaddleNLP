# 1. Copy the python tokenizers directory to the binary directory
add_custom_target(copy_python_tokenizers ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/python ${CMAKE_BINARY_DIR}/python
    DEPENDS core_tokenizers)


# 2. Copy setup.py
add_custom_target(copy_setup ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/setup.py ${CMAKE_BINARY_DIR}/setup.py)

# 3. Copy the core_tokenizers.so to python tokenizers directory
set(TOKENIZER_CORE_NAME "core_tokenizers")
set(TOKENIZER_DST_DIR ${CMAKE_BINARY_DIR}/python/faster_tokenizer)
set(TOKENIZER_SRC_DIR ${CMAKE_BINARY_DIR}/faster_tokenizer/src)
set(TOKENIZER_THIRD_PARTY_DIR ${CMAKE_BINARY_DIR}/third_party)

IF(WIN32)
set(ICU_DLL_DIR ${TOKENIZER_THIRD_PARTY_DIR}/icu/src/extern_icu/icu4c/bin64)
add_custom_target(copy_shared_library ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${TOKENIZER_SRC_DIR}/${TOKENIZER_CORE_NAME}.pyd ${TOKENIZER_SRC_DIR}/${TOKENIZER_CORE_NAME}.lib ${TOKENIZER_DST_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${ICU_DLL_DIR}/icudt70.dll ${ICU_DLL_DIR}/icuuc70.dll ${TOKENIZER_DST_DIR}/libs
    DEPENDS copy_python_tokenizers)
ELSE(WIN32)
add_custom_target(copy_shared_library ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${TOKENIZER_SRC_DIR}/${TOKENIZER_CORE_NAME}.so ${TOKENIZER_DST_DIR}
    DEPENDS copy_python_tokenizers)
ENDIF()

add_custom_target(create_commit_id_file ALL
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%H > ${TOKENIZER_DST_DIR}/commit.log
    DEPENDS copy_python_tokenizers)
