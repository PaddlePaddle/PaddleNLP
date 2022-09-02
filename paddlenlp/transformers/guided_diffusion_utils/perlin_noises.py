'''
Perlin noise implementation by Paddle.
This code is rewritten based on:
https://github.com/jina-ai/discoart/blob/main/discoart/nn/perlin_noises.py
'''
import numpy as np
import paddle
import paddle.vision.transforms as PF
from PIL import Image, ImageOps


def interp(t):
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10):
    gx, gy = paddle.randn([2, width + 1, height + 1, 1, 1])
    xs = paddle.linspace(0, 1, scale + 1)[:-1, None]
    ys = paddle.linspace(0, 1, scale + 1)[None, :-1]
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] *
                                   (1 - ys))
    return dots.transpose([0, 2, 1, 3]).reshape([width * scale, height * scale])


def perlin_ms(octaves, width, height, grayscale):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2**len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return paddle.concat(out_array)


def create_perlin_noise(octaves, width, height, grayscale, side_y, side_x):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = PF.resize(size=(side_y, side_x), img=out.numpy())
        out = np.uint8(out)
        out = Image.fromarray(out).convert('RGB')
    else:
        out = out.reshape([-1, 3, out.shape[0] // 3, out.shape[1]])
        out = out.squeeze().transpose([1, 2, 0]).numpy()
        out = PF.resize(size=(side_y, side_x), img=out)
        out = out.clip(0, 1) * 255
        out = np.uint8(out)
        out = Image.fromarray(out)

    out = ImageOps.autocontrast(out)
    return out


def regen_perlin(perlin_mode, side_y, side_x, batch_size):
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1,
                                   False, side_y, side_x)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4,
                                    False, side_y, side_x)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1,
                                   True, side_y, side_x)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4,
                                    True, side_y, side_x)
    else:
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1,
                                   False, side_y, side_x)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4,
                                    True, side_y, side_x)

    init = (PF.to_tensor(init).add(PF.to_tensor(init2)).divide(
        paddle.to_tensor(2.0)).unsqueeze(0) * 2 - 1)
    del init2
    return init.expand([batch_size, -1, -1, -1])
