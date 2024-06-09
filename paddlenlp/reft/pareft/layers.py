import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LowRankRotateLayer(nn.Layer):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m):
        super().__init__()
        # n > m
        print("n,m", n, m)

        # weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal())
        # linear = paddle.nn.Linear(10, 15, weight_attr=weight_attr)
        self.weight = self.create_parameter(
            shape=[n, m],
            attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal()),
            is_bias=False,
        )
        # print(self.weight.T @ self.weight )

    def forward(self, x):
        return paddle.matmul(x.astype(self.weight.dtype), self.weight)


# # 示例用法
# n, m = 4096, 4  # 示例维度
# layer = LowRankRotateLayer(n, m)
# layer.to("gpu")
# x = paddle.randn([n]).to("gpu")  # 示例输入
# print(x)
# output = layer(x)
# print(output)
