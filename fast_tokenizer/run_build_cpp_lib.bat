if not exist build_cpp mkdir build_cpp
cd build_cpp
for /d %%G in ("*") do rmdir /s /q "%%G"
del /q *
cmake .. -G "Ninja" -DWITH_PYTHON=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
ninja -j20
cd ..