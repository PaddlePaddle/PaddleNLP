from scipy.optimize import curve_fit

def nihe():
    x_data = [1, 2, 3, 4, 5, 6]
    y_data = [
                371.8363922,
                742.1444031,
                1090.63053,
                1467.008313,
                1833.177271,
                2199.174609
            ]
    def linear_func(x, m, c):
        return m * x + c
    popt, _ = curve_fit(linear_func, x_data, y_data)
    
    def static_func(x):
        return x * 371.78662414550786
    
    test_bsz_list = list(range(1, 32))
    a_res, b_res = [], []
    for bsz in test_bsz_list:
        result_a = linear_func(bsz, *popt)
        result_b = static_func(bsz)
        a_res.append(result_a)
        b_res.append(result_b)
        delta = result_a - result_b
        percent_a = delta / result_a
        percent_b = delta / result_b
        print(f'bsz{bsz}, result_a:{result_a},  result_b:{result_b}, delta:{delta}, percent_a:{percent_a}, percent_b:{percent_b}') # 最终的percent在1-2%左右
    
if __name__ == '__main__':
    nihe()