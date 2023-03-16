from paddlenlp.layers import LoRAConfig, LoRALinear, get_lora_model
from paddlenlp.transformers import AutoModel
lora_config = LoRAConfig(
    target_modules=[".*q_proj.*", ".*v_proj.*"],
    r=4,
    lora_alpha=8,
    merge_weights=True,
)
model = AutoModel.from_pretrained("ernie-3.0-nano-zh")
lora_model = get_lora_model(model, lora_config)