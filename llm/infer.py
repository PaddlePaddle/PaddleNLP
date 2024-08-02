from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

def infer(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors='pd')
    outputs = model.generate(**inputs, decode_strategy='greedy_search')
    return tokenizer.decode(outputs[0][0], skip_special_tokens=True)

path1="checkpoints/ckpt_quant_pt_1000/checkpoint-1000-ori"
path2="checkpoints/ckpt_quant_pt_1000/checkpoint-1000"

tokenizer1 = AutoTokenizer.from_pretrained(path1)
tokenizer2 = AutoTokenizer.from_pretrained(path2)

model1 = AutoModelForCausalLM.from_pretrained(path1)
model2 = AutoModelForCausalLM.from_pretrained(path2)

input_list = [
    "hello, how are you today?",
    "who is the president of America?",
    "类型#裙*版型#显瘦*风格#民族风*图案#刺绣*裙长#长裙*裙款式#勾花镂空",
    "类型#裙*颜色#黑白*风格#复古*风格#知性*图案#条纹*图案#蝴蝶结*图案#复古*裙长#连衣裙*裙款式#不规则*裙款式#收腰",
]

for query in input_list:
    output1 = infer(model1, tokenizer1, query)
    output2 = infer(model2, tokenizer2, query)
    print("=="*30)
    print(f"model1 output: {output1}")
    print(f"model2 output: {output2}")


