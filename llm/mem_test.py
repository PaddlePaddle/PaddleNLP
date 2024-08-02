import sys
import gc
import copy
a = {}
print("init empty dict memory size={} bytes".format(sys.getsizeof(a)))

for i in range(10**6):
    a[i] = i
print("after set value, dict memory size={} bytes".format(sys.getsizeof(a)))

#for i in range(10**6):
#    del a[i]
#    # a.pop(i)

del a
gc.collect()

print("after del, dict memory size={} bytes".format(sys.getsizeof(a)))
a_new = dict(a)
print("after init a new one, dict memory size={} bytes".format(sys.getsizeof(a_new)))
b = copy.copy(a)
print("after copy a new one, dict memory size={} bytes".format(sys.getsizeof(b)))
c = copy.deepcopy(a)
print("after deepcopy a new one, dict memory size={} bytes".format(sys.getsizeof(c)))

