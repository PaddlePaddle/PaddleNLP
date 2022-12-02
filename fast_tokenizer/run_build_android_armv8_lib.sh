mkdir build_android_arm64_v8a
cd build_android_arm64_v8a
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_NATIVE_API_LEVEL=android-21 -DANDROID_STL=c++_static -DWITH_TESTING=OFF -DWITH_PYTHON=OFF -DANDROID_TOOLCHAIN=clang  
make -j8
