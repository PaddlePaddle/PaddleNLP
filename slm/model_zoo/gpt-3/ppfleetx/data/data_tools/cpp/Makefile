CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell $(PYTHON_BIN) -m pybind11 --includes)
CPPFLAGS += $(shell python3-config --includes)

LIBNAME = fast_index_map_helpers
LIBEXT = .so

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
