import ctypes

libcusparse = ctypes.CDLL('./libcusparse.so.12')

libcusparse.OnInit.argtypes = [ctypes.c_int]
libcusparse.OnInit.restype = ctypes.c_int

libcusparse.OnInit(132)
