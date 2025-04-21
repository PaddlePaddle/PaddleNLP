import paddle
from typing import List, Optional
from paddle.autograd import PyLayer
from .int8_kernel import per_token_group_quant_int8, w8a8_block_int8_matmul, create_int8_parameter

def int8_linear(
    input: paddle.Tensor,
    weight: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
    weight_scale: Optional[paddle.Tensor] = None,
    weight_dtype: Optional[paddle.dtype] = None,
    block_size: Optional[int] = None,
):
    """
    INT8 Linear 替代函数，封装为 PyLayer 以支持动态图兼容性。

    Args:
        x (Tensor): 输入张量 [B, ..., input_dim]
        weight (Tensor): INT8 权重张量 [output_dim, input_dim]
        bias (Tensor, optional): 偏置张量 [output_dim]
        weight_scale (Tensor, optional): 权重量化 scale
        weight_dtype (paddle.dtype, optional): 权重的原始数据类型
        group_size (int, optional): 分组量化的大小（对应 block_size 的 K 维度）

    Returns:
        Tensor: 经过 INT8 推理后的输出张量
    """
    return Int8Linear.apply(input, weight, bias, block_size, weight_scale, None)
        
class Int8Linear(PyLayer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        block_size: List[int],
        dtype: paddle.dtype = paddle.float32,
        bias: bool = True,
        quant_config: Optional[dict] = None,
    ):
        super().__init__()
        if len(block_size) != 2:
            raise ValueError("block_size 必须是一个包含两个整数的列表")
        self.input_size = input_size
        self.output_size = output_size
        self.block_size = block_size
        self.dtype = dtype
        self.bias = bias
        self.quant_config = quant_config or {"is_checkpoint_int8_serialized": True, "activation_scheme": "dynamic"}

        block_n, block_k = self.block_size
        self.weight_scale = paddle.create_parameter(
            shape=[self.output_size // block_n],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.create_weights()

        if bias:
            self.bias = paddle.create_parameter(
                shape=[self.output_size],
                dtype=self.dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
                is_bias=True,
            )
        else:
            self.bias = None
        
    def create_weights(self):
        block_n, block_k = self.block_size

        if self.input_size % block_k != 0:
            raise ValueError(f"input_size {self.input_size} must be divisible by block_k {block_k}")
        
        if self.output_size % block_n != 0:
            raise ValueError(f"output_size {self.output_size} must be divisible by block_n {block_n}")
        
        weight_dtype = (
            paddle.int8
            if self.quant_config.get("is_checkpoint_int8_serialized", False)
            else self.dtype
        )
        if weight_dtype == paddle.int8:
            self.weight = create_int8_parameter(self, "weight", [self.input_size, self.output_size])
        else:
            self.weight = paddle.create_parameter(
                shape=[self.input_size, self.output_size],
                dtype=weight_dtype,
                is_bias=False,
            )
        self.weight_scale.set_value(paddle.full(self.weight_scale.shape, paddle.finfo(paddle.float32).min))
        assert self.quant_config.get("activation_scheme", "dynamic") == "dynamic"
        self.input_scale = None
        
    #根据quick lora实现
    @staticmethod
    def forward(
        ctx,
        input: paddle.Tensor,
        weight: paddle.Tensor,
        bias: Optional[paddle.Tensor],
        block_size: List[int],
        weight_scale: paddle.Tensor,
        input_scale: Optional[paddle.Tensor],
    ):
        input_shape = input.shape # [bsz, qlen, hidden_dim] [1, 7, 896]
        input_2d = input.reshape([-1, input.shape[-1]])
        output_shape = [*input.shape[:-1], weight.shape[0]]

        # Step 1: Per-token group quantization
        q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])
        # q_input.shape [7, 896]
        # x_scale.shape [7, 14]
        # Step 2: Int8 matmul + dequant
        # weight_scale = weight_scale.transpose([1, 0])
        output = w8a8_block_int8_matmul(q_input, weight, x_scale, weight_scale, block_size, output_dtype=input.dtype)
        if bias is not None:
            output = output + bias.astype(output.dtype)
        
        output = output.reshape(output_shape)

        ctx.save_for_backward(input, weight, q_input, x_scale, weight_scale,)
        ctx.block_size = block_size
        ctx.bias = bias
        ctx.input_shape = input_shape
        ctx.input_stop_gradient = input.stop_gradient
        ctx.weight_stop_gradient = weight.stop_gradient
        ctx.bias_stop_gradient = bias.stop_gradient if bias is not None else True

        return output
    
    @staticmethod
    def backward(
        ctx,
        grad_output: paddle.Tensor,
    ):
        input, weight, q_input, x_scale, weight_scale = ctx.saved_tensor()
        block_size = ctx.block_size
        bias = ctx.bias
        input_shape = ctx.input_shape

        grad_input = grad_weight = grad_bias = None
        grad_output_2d = grad_output.reshape([-1, grad_output.shape[-1]])


        if not ctx.input_stop_gradient:
            grad_input_2d = paddle.matmul(grad_output, weight, transpose_y=True)
            grad_input = grad_input_2d.reshape(input_shape)
        
        if not ctx.weight.stop_gradient:
            grad_weight = paddle.matmul(grad_output_2d, q_input, transpose_x=True) * x_scale.unsqueeze(-1)

        if bias is not None and not bias.stop_gradient:
            grad_bias = grad_output.sum(axis=tuple(range(len(grad_output.shape) - 1)))

        return grad_input, grad_weight, grad_bias, None, None, None

# if __name__ == "__main__":
#     layer = Int8Linear(
#         input_size=1024,
#         output_size=512,
#         block_size=[128, 128],
#         dtype=paddle.float32,
#         bias=True,
#     )
#     print("Weight shape:", layer.weight.shape)
#     print("Weight scale shape:", layer.weight_scale.shape)
#     print("Bias shape:", layer.bias.shape if layer.bias is not None else None)
    


    
        
        

    
        
        
                


            
            


        
            