import re
import json

# 从文件中读取文本
with open('res.txt', 'r') as f:
    text = f.read()

# 使用正则表达式匹配所有参数
pattern = r'pp_size=(\d+),\s*tp_size=(\d+),\s*use_ulysses=(\d+),\s*dp_size=(\d+),\s*sharding_stage=(\d+),\s*recompute=(\d+)'
matches = re.findall(pattern, text)

pp_size_list = [int(match[0]) for match in matches]
tp_size_list = [int(match[1]) for match in matches]
use_ulysses_list = [int(match[2]) for match in matches]
dp_size_list = [int(match[3]) for match in matches]
sharding_stage_list = [int(match[4]) for match in matches]
recompute_list = [int(match[5]) for match in matches]

pp_stage_idx_list = [0 for _ in range(len(pp_size_list))]

res = {
    'pp_size': pp_size_list[0],
    'dp_size_list': ','.join(map(str, dp_size_list)),
    'sharding_stage_list': ','.join(map(str, sharding_stage_list)),
    'tp_size_list': ','.join(map(str, tp_size_list)),
    'usp_flag_list': ','.join(map(str, use_ulysses_list)),
    'pp_stage_idx_list': ','.join(map(str, pp_stage_idx_list)),
    'recompute_list': ','.join(map(str, recompute_list)),
    'vtp': tp_size_list[0],
    'vsp_flag': 0,
    'embed_sdp': 1
}

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4)  # indent=4 使 JSON 格式化，便于阅读




# # 将结果转换为字典列表
# result = []
# for match in matches:
#     result.append({
#         'pp_size': int(match[0]),
#         'tp_size': int(match[1]),
#         'use_ulysses': int(match[2]),
#         'dp_size': int(match[3]),
#         'sharding_stage': int(match[4]),
#         'recompute': int(match[5])
#     })

# # 打印结果
# for i, params in enumerate(result, 1):
#     print(f"Layer {i}: {params}")

# # 保存到文件
# with open('extracted_parameters.txt', 'w') as f:
#     for i, params in enumerate(result, 1):
#         f.write(f"Layer {i}: {params}\n")