"""
Helpers for various likelihood-based losses implemented by Paddle. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""
import paddle
import paddle.nn.functional as F


def spherical_dist_loss(x, y):
    x = F.normalize(x, axis=-1)
    y = F.normalize(y, axis=-1)
    return (x - y).norm(axis=-1).divide(
        paddle.to_tensor(2.0)).asin().pow(2).multiply(paddle.to_tensor(2.0))


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clip(-1, 1)).pow(2).mean([1, 2, 3])
