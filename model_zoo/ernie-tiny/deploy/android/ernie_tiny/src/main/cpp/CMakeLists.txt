# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.
project("ernie_tiny_jni")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

set(FastDeploy_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/fastdeploy-android-latest-shared-dev")

find_package(FastDeploy REQUIRED)

# include_directories(.)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${FastDeploy_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Ofast -DNDEBUG -fomit-frame-pointer -fno-asynchronous-unwind-tables -fno-unwind-tables")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden -fdata-sections -ffunction-sections")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,-z,nocopyreloc")

add_library(
        ernie_tiny_jni
        SHARED
        ernie_tiny_jni/runtime_option_jni.cc
        ernie_tiny_jni/predictor_jni.cc
)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You can link
# multiple libraries, such as libraries you define in this build script,
# prebuilt third-party libraries, or system libraries.

target_link_libraries(
        # Specifies the target library.
        ernie_tiny_jni
        ${FASTDEPLOY_LIBS}
        ${log-lib}
)
