"""serialization utils"""


def to_dict_with_sorted_values(d, key=None):
    """to dict with sorted values"""
    return {k: sorted(v, key=key) for k, v in d.items()}


def to_dict_with_set_values(d):
    """to dict with set values"""
    result = {}
    for k, v in d.items():
        hashable_v = []
        for v_elem in v:
            if isinstance(v_elem, list):
                hashable_v.append(tuple(v_elem))
            else:
                hashable_v.append(v_elem)
        result[k] = set(hashable_v)
    return result


def tuplify(x):
    """tuplify"""
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(tuplify(elem) for elem in x)
