from operator import imod
import numpy as np
import paddle
from scipy.linalg import block_diag

def process_example(data, tokenizer, data_args):
    """Convert raw format example to Example."""
    if isinstance(data["src"], str):
        data["src"] = [data["src"]]
    if isinstance(data["tgt"], str):
        data["tgt"] = [data["tgt"]]
    if len(data["src"]) != len(data["tgt"]) + 1:
        return None
    if (
        (len(data["response"]) != 2)
        or (len(data["response"]) != len(data["sort"]))
        or data["sort"][0] == data["sort"][1]
    ):
        return None
    if data["sort"][0] > data["sort"][1]:
        chosen = data["response"][0]
        rejected = data["response"][1]
    else:
        chosen = data["response"][1]
        rejected = data["response"][0]

    #example = {"src": data["src"], "tgt": data["tgt"], "chosen": chosen, "rejected": rejected}
    conversations = []
    for idx in range(len(data['src'])):
        if idx < len(data['tgt']):
            conversations.append([
                data['src'][idx].strip(),
                data['tgt'][idx].strip(),
            ])
        else:
            conversations.append([
                data['src'][idx].strip(),
                chosen.strip(),
            ])

    # chat template load
    #llama_chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
    #qwen_chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    #tokenizer.init_chat_template(qwen_chat_template)

    chosen_encode_tokens = tokenizer.encode_chat_inputs(conversations)['conversations']

    # convert to rejected conversations
    conversations[-1][-1] = rejected.strip()

    try:
        rejected_encode_tokens = tokenizer.encode_chat_inputs(conversations)['conversations']
    except:
        print(conversations)
    

    """Post process sequence: tokenization & truncation."""
    tokens_prompt = chosen_encode_tokens[-1][0]
    tokens_chosen = chosen_encode_tokens[-1][-1]
    tokens_rejected = rejected_encode_tokens[-1][-1]
    del rejected_encode_tokens

    # temporary reserve 1 token for truncation
    data_args.max_seq_length = data_args.max_seq_length - 1

    if len(tokens_prompt) + len(tokens_chosen) + len(tokens_rejected) > data_args.max_seq_length:
        # truncate prompt
        tokens_prompt = tokens_prompt[-data_args.max_prompt_len:]
        if (len(tokens_prompt) + len(tokens_chosen) + len(tokens_rejected)) > data_args.max_seq_length:
            max_response_len = (
                data_args.max_seq_length
                - len(tokens_prompt)
            )
            # 按比例截断
            max_chosen_len = int(
                len(tokens_chosen) / (len(tokens_chosen) + len(tokens_rejected)) * max_response_len
            )
            max_rejected_len = max_response_len - max_chosen_len
            tokens_chosen = tokens_chosen[:max_chosen_len]
            tokens_rejected = tokens_rejected[:max_rejected_len]

    cur_len = (
        + len(tokens_prompt)
        + len(tokens_chosen)
        + len(tokens_rejected)
    )
    turn_index = len(chosen_encode_tokens) - 2

    # append former dialog contents
    while turn_index >= 0:
        tokens_src = chosen_encode_tokens[turn_index][0]
        tokens_target = chosen_encode_tokens[turn_index][1]
        turn_index -= 1

        if len(tokens_src) + len(tokens_target) > data_args.max_seq_length - cur_len:
            break
        tokens_prompt = tokens_src + tokens_target + tokens_prompt
        cur_len += len(tokens_src) + len(tokens_target)
    
    concatenated_input_ids = (
        tokens_prompt
        + tokens_chosen
        + [tokens_prompt[-1]]
        + tokens_rejected
    )

    # make position ids & labels
    prompt_len = len(tokens_prompt)
    concatenated_position_ids = (
        list(range(prompt_len))
        + list(range(prompt_len, prompt_len + len(tokens_chosen))) 
        + list(range(prompt_len, prompt_len + len(tokens_rejected) + 1))
    )

    chosen_labels = (
        [0] * len(tokens_prompt))
    chosen_labels += tokens_chosen
    chosen_labels += [0] * (len(tokens_rejected) + 1)

    rejected_labels = (
        [0] * len(tokens_prompt))
    #rejected_labels += [tokens_rejected[0]]
    rejected_labels += [0] * len(tokens_chosen)
    rejected_labels += [tokens_prompt[-1]]
    rejected_labels += tokens_rejected

    # shift labels
    concatenated_input_ids = concatenated_input_ids[:-1]
    concatenated_position_ids = concatenated_position_ids[:-1]
    chosen_labels = chosen_labels[1:]
    rejected_labels = rejected_labels[1:]

    seq_len = len(concatenated_input_ids)

    # ?
    if seq_len > data_args.max_seq_length:
        return None

    concatenated_attention_mask = np.tri(seq_len, seq_len, dtype=bool)
    chosen_input_ids_start_index = prompt_len
    rejected_input_ids_start_index = prompt_len + len(tokens_chosen)

    concatenated_attention_mask[
        rejected_input_ids_start_index:, chosen_input_ids_start_index: rejected_input_ids_start_index
    ] = False

    chosen_labels_start_index = len(tokens_prompt)
    rejected_labels_start_index = chosen_labels_start_index  + len(tokens_chosen)
    response_index = [chosen_labels_start_index, rejected_labels_start_index, seq_len]

    # undo change
    data_args.max_seq_length = data_args.max_seq_length + 1

    return {
        "input_ids": concatenated_input_ids,
        "position_ids": concatenated_position_ids,
        "attention_mask": concatenated_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_labels": rejected_labels,
        "response_index": response_index,
    }
    

def collate_fn(batch, tokenizer, max_seq_length=None):
    """Convert batch data into tensor."""
    # max_seq_length = 4096
    if max_seq_length is None:
        raise ValueError("max_seq_length is None.")

    input_dict = {
        "input_ids": [],
        "position_ids": [],
        "attention_mask": [],
        "chosen_labels": [],
        "rejected_labels": [],
        "response_indexs": [],
    }
    for i, sequence in enumerate(batch):
        difference = max_seq_length - len(sequence['input_ids'])

        input_dict["input_ids"].append(
            sequence["input_ids"]
            + [tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id] * difference
        )
        input_dict["position_ids"].append(
            sequence["position_ids"] + [0] * difference
        )
        input_dict["chosen_labels"].append(
            sequence["chosen_labels"] + [0] * difference
        )
        input_dict["rejected_labels"].append(
            sequence["rejected_labels"] + [0] * difference
        )

        for ri in sequence["response_index"]:
            input_dict["response_indexs"].append([
                i,      # bs
                ri[0],  # chosen_response_start_index
                ri[1],  # rejeted_response_start_index
                ri[2],  # rejeted_response_end_index + 1
            ])

        input_dict["attention_mask"].append(
            # pad to max_loength
            np.pad(
                sequence["attention_mask"],
                pad_width=((0, 0), (0, difference), (0, difference)),
                mode="constant",
                constant_values=False,
            )
        )
    for key in input_dict:
        input_dict[key] = paddle.to_tensor(input_dict[key])
    input_dict["attention_mask"] = input_dict["attention_mask"].cast("float32")
    return input_dict