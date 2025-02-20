

import paddle
import paddle.nn.functional as F
import numpy as np
import unittest
from typing import Dict, Any, List, Tuple
from parameterized import parameterized

from paddlenlp_ops import trt_llm_fused_moe
import numpy as np

class TestMoe(unittest.TestCase):
    """Test class for MoE (Mixture of Experts) model"""

    # Test configuration parameters
    DEFAULT_TEST_CONFIGS = {
        # 'rows': [2, 16, 512, 2048],
        # 'ks': [2, 4],
        # 'experts_list': [32],
        # 'hidden_sizes': [1024, 2048],
        # 'inter_sizes': [4096],
        'rows': [1],
        'ks': [4],
        'experts_list': [64],
        'hidden_sizes': [2048],
        'inter_sizes': [1408],
    }

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize test class setup"""
        paddle.seed(2)

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean: float = 0, std: float = 1):
        # return paddle.randn(shape, dtype=dtype) / 10
        
        return paddle.empty(shape, dtype=dtype).normal_(mean, std)

    @staticmethod
    def basic_moe_fc(activations, expert_for_row, weights):
        
        res = paddle.zeros(shape=[activations.shape[0], weights.shape[-1]], 
                         dtype=activations.dtype)
        
        for row in range(activations.shape[0]):
            row_expert = expert_for_row[row]
            # print(activations[row].shape)
            # print(weights[row_expert].shape)
            out  = paddle.matmul(activations[row], weights[row_expert])
            # print(out.shape)
            res[row : row + 1, :] = out
        return res

    @staticmethod
    def apply_activation(inp, act_str):
        """Apply activation function

        Args:
            inp: Input tensor
            act_str: Activation function name

        Returns:
            torch.Tensor: Activated tensor
        """
        activation_map = {
            # "identity": lambda x: x,
            # "silu": torch.nn.SiLU(),
            # "relu": torch.nn.ReLU(),
            # "gelu": lambda x: torch.nn.GELU(approximate="tanh")(x),
            "swiglu": paddle.incubate.nn.functional.Swiglu()
        }
        
        if act_str not in activation_map:
            raise ValueError(f"Unsupported activation: {act_str}")
            
        return activation_map[act_str](inp)

    def generate_inputs(self, num_rows: int, active_rows: int, 
                       hidden_size: int, num_experts: int, 
                       dtype, quant_type):
       
        return {
            "input_activations": self.create_random_cuda_tensor([num_rows, hidden_size], dtype, mean=0, std=0.01),
            "gating_output": self.create_random_cuda_tensor([num_rows, num_experts], dtype).cast("float32")
        }

    def generate_weights(self, hidden_size: int, inter_size: int, 
                        num_experts: int, dtype, 
                        quant_type):
        
        weights = {}
        for prefix in ['fc1', 'fc2']:
            if prefix == 'fc1':
                # shape = [num_experts, hidden_size, inter_size * 2]
                shape = [num_experts, hidden_size, inter_size]
            else:
                shape = [num_experts, inter_size, hidden_size]
                
            ref_weights = self.create_random_cuda_tensor(shape, dtype, mean=0, std=0.01)
            weights[f'{prefix}_expert_weights_for_ref'] = ref_weights
            # Cutlass GEMM的版本要求的权重是Col Major的
            weights[f'{prefix}_expert_weights_for_ft'] = paddle.transpose(ref_weights, perm=[0, 2, 1]).reshape(shape)
            
            
        return weights

    def run_ft_moe(self, input_dict, 
                   active_rows: int, k: int, activation_str: str):

        return trt_llm_fused_moe(
            input_dict["input_activations"],
            input_dict["gating_output"],
            input_dict["fc1_expert_weights_for_ft"],
            input_dict["fc2_expert_weights_for_ft"],
            None,
            None,
            None,
            k,
            3,
            "none"
        )

    def run_ref_moe(self, input_dict, 
                    k, activation_str) :

        input_dict["gating_output"] = input_dict["gating_output"].cast(paddle.float32)
        gates = F.softmax(input_dict["gating_output"], axis=-1).cast(paddle.float32)

        expert_scales, experts_for_row = paddle.topk(gates, k, axis=-1, sorted=False)
        expert_scales /= expert_scales.sum(axis=-1, keepdim=True)

        output = paddle.zeros_like(input_dict["input_activations"])
        
        for k_idx in range(k):
            current_expert_scales = expert_scales[:, k_idx].unsqueeze(1)
            current_experts_for_row = experts_for_row[:, k_idx]

            fc1_out = self.basic_moe_fc(
                input_dict["input_activations"],
                current_experts_for_row,
                input_dict["fc1_expert_weights_for_ref"]
            )
            
            activated = paddle.incubate.nn.functional.swiglu(fc1_out)
            # activated = paddle.nn.functional.silu(fc1_out)

            fc2_out = self.basic_moe_fc(
                activated,
                current_experts_for_row,
                input_dict["fc2_expert_weights_for_ref"]
            )
            # print(current_expert_scales * fc2_out)
            output += current_expert_scales * fc2_out
    
        return output

    def run_moe_test(self, dtype, quant_type,
                     rtol: float, atol: float, activation_str: str = "gelu",
                     test_configs: Dict[str, List] = None) -> None:
        """Run MoE test

        Args:
            dtype: Data type
            quant_type: Quantization type
            rtol: Relative tolerance
            atol: Absolute tolerance
            activation_str: Activation function name
            test_configs: Test configuration parameters
        """
        paddle.device.cuda.empty_cache()
        
        if test_configs is None:
            test_configs = self.DEFAULT_TEST_CONFIGS
            
        for hidden_size in test_configs['hidden_sizes']:
            for inter_size in test_configs['inter_sizes']:
                for experts in test_configs['experts_list']:
                    weights = self.generate_weights(hidden_size, inter_size, experts, dtype, quant_type)
                    
                    for row in test_configs['rows']:
                        for k in test_configs['ks']:
                            if k > experts:
                                continue
                                
                            input_dict = self.generate_inputs(row, row, hidden_size, experts, dtype, quant_type)
                            input_dict.update(weights)
                            
                            act_output = self.run_ft_moe(input_dict, row, k, activation_str).cast(np.float32)
                            print(act_output)

                            # ref_output = self.run_ref_moe(input_dict, k, activation_str).cast(np.float32)
                            # print(ref_output)
                            # print("done !----------------------------------")

                            # print(act_output - ref_output)
                            # print(paddle.max(paddle.abs(act_output - ref_output)))

                            # np.testing.assert_allclose(
                            #     act_output, ref_output, rtol=rtol, atol=atol
                            # )

    @parameterized.expand([
        ("fp32_swilu", paddle.float32, paddle.float32, 1e-2, 1e-2),
        ("fp16_swilu", paddle.float16, paddle.float16, 1e-2, 1e-2),
        # ("bf16_silu", paddle.bfloat16, paddle.bfloat16, 1e-2, 1e-2),
    ])
    def test_moe(self, name: str, dtype, quant_type,
                 rtol: float, atol: float) -> None:
        """Parameterized MoE test

        Args:
            name: Test name
            dtype: Data type
            quant_type: Quantization type
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.run_moe_test(
            dtype=dtype,
            quant_type=quant_type,
            rtol=rtol,
            atol=atol,
            # activation_str="silu",
            activation_str="Swiglu"
        )

if __name__ == '__main__':
    unittest.main()
