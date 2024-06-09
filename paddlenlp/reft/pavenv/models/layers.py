import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class InverseRotateLayer(nn.Layer):
    """The inverse of a given `LinearLayer` module."""

    def __init__(self, lin_layer):
        super(InverseRotateLayer, self).__init__()
        self.lin_layer = lin_layer

    def forward(self, x):
        output = paddle.matmul(x, self.lin_layer.weight)
        return output


class RotateLayer(nn.Layer):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super(RotateLayer, self).__init__()
        weight = paddle.empty([n, n], dtype="float32")
        if init_orth:
            paddle.nn.initializer.Orthogonal()(weight)
        self.weight = self.create_parameter(
            shape=weight.shape, default_initializer=paddle.nn.initializer.Assign(weight)
        )

    def forward(self, x):
        return paddle.matmul(x.astype(self.weight.dtype), self.weight)


class LowRankRotateLayer(nn.Layer):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m):
        super(LowRankRotateLayer, self).__init__()
        self.weight = self.create_parameter(
            shape=[n, m], default_initializer=paddle.nn.initializer.Orthogonal()
        )

    def forward(self, x):
        return paddle.matmul(x.astype(self.weight.dtype), self.weight)


class SubspaceLowRankRotateLayer(nn.Layer):
    """A linear transformation with orthogonal initialization with subspace."""

    def __init__(self, n, m):
        super(SubspaceLowRankRotateLayer, self).__init__()
        self.weight = self.create_parameter(
            shape=[n, m], default_initializer=paddle.nn.initializer.Orthogonal()
        )

    def forward(self, x, l, r):
        return paddle.matmul(x.astype(self.weight.dtype), self.weight[:, l:r])
