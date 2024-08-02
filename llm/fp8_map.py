import itertools
import paddle

def create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e+p == total_bits-has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
    pvalues = []
    for i, val in enumerate(range(-((2**(exponent_bits-has_sign))), 2**(exponent_bits-has_sign), 1)):
        evalues.append(2**val)
    print(evalues)


    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    print(lst)
    #for ev in evalues:
    bias = 2**(exponent_bits-1)
    for evalue in range(2**(exponent_bits)):
        for bit_pattern in lst:
            value = (1 if evalue != 0 else 0)
            for i, pval in enumerate(list(bit_pattern)):
                value += pval*(2**-(i+1))
            print(evalue, bit_pattern, value)
            if evalue == 0:
                # subnormals
                value = value*2**-(bias)
            else:
                # normals
                value = value*2**-(evalue-bias-1)
            print('n :', value)
            values.append(value)
            if signed:
                values.append(-value)


    assert len(values) == 2**total_bits
    values.sort()
    print(values)
    if total_bits < 8:
        gap = 256 - len(values)
        for i in range(gap):
            values.append(0)
    values.sort()
    code = paddle.to_tensor(values)
    code /= code.max()

    return code

code = create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4)
#code = create_fp8_map()
import pdb; pdb.set_trace()
