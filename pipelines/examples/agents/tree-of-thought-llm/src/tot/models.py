# coding=utf8, ErnestinaQiu
from llama2 import ChatCompletion

completion_tokens = prompt_tokens = 0


def completions_with_backoff(**kwargs):
    chatter = kwargs["chatter"]
    print(kwargs["messages"])
    return chatter.create(messages=kwargs["messages"], temperature=kwargs["temperature"])


def gpt(prompt, model="llama-2-7b-chat", temperature=0.6, max_tokens=1000, n=1, stop=None) -> list:
    messages = [[{"role": "system", "content": ""}, {"role": "user", "content": prompt}]]
    return chatgpt(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, n=1, stop=stop)


def chatgpt(messages, model="llama-2-7b-chat", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    if model == "llama-2-7b-chat":
        chatter = ChatCompletion()
    elif model == "llama2-13b-chat":
        chatter = ChatCompletion(model)
    elif model == "llama2-70b-chat":
        chatter = ChatCompletion(model)
    else:
        print(f"Not support for llm {model}, and use llama-2-7b-chat instead.")
        chatter = ChatCompletion()

    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        print("messages: {}".format(messages))
        res = completions_with_backoff(chatter=chatter, model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs

    
def gpt_usage(backend="llama-2-7b-chat"):
    global completion_tokens, prompt_tokens
    cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
