from .lora_model import LoRAModel
from .lora_layers import LoRALinear

import paddle
from paddlenlp.quantization.qlora import qlora_weight_quantize_dequantize


def transform_lora_layers(
    model: LoRAModel,
    num_iterations: int = 100
) -> None:
    if not isinstance(model, LoRAModel):
        raise NotImplementedError(f"Unknown model type: {type(model)}")
    for name, submodule in model.named_sublayers():
        if isinstance(submodule, LoRALinear):
            num_ranks = submodule.r
            W = submodule.weight

            if W.dtype in [paddle.float16]:
                old_dtype = W.dtype
                W = paddle.cast(W, dtype=paddle.float32)
            else:
                old_dtype = None

            Q = paddle.zeros_like(W)
            last_error = paddle.to_tensor(float("inf"), dtype=W.dtype)
            for i in range(num_iterations):
                A = W - Q
                if A.ndim != 2:
                    raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")

                U, S, Vh = paddle.linalg.svd(A, full_matrices=False)
                Ur = U[:, :num_ranks]
                Sr = S[:num_ranks]
                Vhr = Vh[:num_ranks]
                
                lora_A = Ur @ paddle.diag(paddle.sqrt(Sr))
                lora_B = paddle.diag(paddle.sqrt(Sr)) @ Vhr
                
                Q = qlora_weight_quantize_dequantize(W-lora_A@lora_B, double_quant=True)
                
                W_ = Q + lora_A@lora_B
                error = paddle.norm(W - W_, p = "fro")

                if error > last_error:
                    print("break.")
                    break
                last_error = error

            if old_dtype is not None:
                lora_A = paddle.cast(lora_A, dtype=old_dtype)
                lora_B = paddle.cast(lora_B, dtype=old_dtype)
                Q = paddle.cast(Q, dtype=old_dtype)

            submodule.lora_A.set_value(lora_A)
            submodule.lora_B.set_value(lora_B)
            submodule.weight.set_value(Q)