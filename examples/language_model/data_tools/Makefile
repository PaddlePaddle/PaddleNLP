CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell /usr/bin/python3.8 -m pybind11 --includes)
LIBNAME = helpers
LIBEXT = $(shell /usr/bin/python3.8-config --extension-suffix)

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
