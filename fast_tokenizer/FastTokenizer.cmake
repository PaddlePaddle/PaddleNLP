CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-narrowing")
else()
    set(CMAKE_CXX_STANDARD 11)
endif()

set(LIBRARY_NAME core_tokenizers)

set(FAST_TOKENIZER_INCS "")
list(APPEND FAST_TOKENIZER_INCS ${CMAKE_CURRENT_LIST_DIR}/include)
list(APPEND FAST_TOKENIZER_INCS ${CMAKE_CURRENT_LIST_DIR}/third_party/include)

set(FAST_TOKENIZER_LIBS "")
find_library(FTLIB ${LIBRARY_NAME} ${CMAKE_CURRENT_LIST_DIR}/lib NO_DEFAULT_PATH)
list(APPEND FAST_TOKENIZER_LIBS ${FTLIB})

if (WIN32)
find_library(ICUDT icudt ${CMAKE_CURRENT_LIST_DIR}/third_party/lib NO_DEFAULT_PATH)
list(APPEND FAST_TOKENIZER_LIBS ${ICUDT})

find_library(ICUUC icuuc ${CMAKE_CURRENT_LIST_DIR}/third_party/lib NO_DEFAULT_PATH)
list(APPEND FAST_TOKENIZER_LIBS ${ICUUC})
endif()
