for %%x in (6 7 8 9 10) do (
  if not exist build_py3%%x mkdir build_py3%%x
  cd build_py3%%x
  for /d %%G in ("*") do rmdir /s /q "%%G"
  del /q *
  cmake .. -G "Ninja" -DWITH_PYTHON=ON ^
                      -DWITH_TESTING=OFF ^
                      -DCMAKE_BUILD_TYPE=Release ^
                      -DPYTHON_EXECUTABLE=C:\Python3%%x\python.exe ^
                      -DPYTHON_INCLUDE_DIR=C:\Python3%%x\include ^
                      -DPYTHON_LIBRARY=C:\Python3%%x\libs\python3%%x.lib
  ninja -j20
  cd ..
)
