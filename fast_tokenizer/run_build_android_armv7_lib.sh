mkdir build_android_armeabi_v7a
cd build_android_armeabi_v7a
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_ROOT/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_NATIVE_API_LEVEL=android-21 -DANDROID_STL=c++_static -DWITH_TESTING=OFF -DWITH_PYTHON=OFF -DANDROID_TOOLCHAIN=clang  
make -j8
