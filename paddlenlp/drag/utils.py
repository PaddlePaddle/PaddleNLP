import jsonlines
import json
import copy
import re

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": [{"role":"system", "content":"Write a response that appropriately completes the request."},
        {"role":"user", "content":"### Instruction:\n{instruction}\n\n### Response:\n"}             
    ],
    "prompt_no_input_retrieval": [{"role":"system", "content":
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"},
        {"role":"user", "content":"### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"}
    ],
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
    "prompt_for_triplet": [{"role":"system", "content":
        "You will receive an instruction ." 
        "Analyze the instruction and extracted two specific types of information:" 
        "1.Subject: The main entity or instance mentioned in the question. "
        "This could be a person, object, organization, event, or activity, etc. It may consist of one or more words."
        "2.Relationship:  The summarization of the relative relationship to the subject and the expected answer" #and also reflect the nature of the expected answer.
        "Summarize explicit relationship information from the input instruction. "
        "Do not add additional guesses about the nature of the subject, but do not omit information explicitly mentioned in the original instruction. It may consist of one or more words.\n\n"   
        "Provide the output in the following format:\n"
        "Subject: XXX\n"
        "Relationship: XXX\n"
        "Explanation: XXX\n"
        "END\n\n"
        },
        {"role":"user", "content":
        "### Instruction: \n{instruction}\n"}
    ],
    "prompt_for_same": [{"role":"system", "content": "You will receive a paragraph and some information (including a subject and a relationship) extracted from a query."
        "Firstly, determine if the subject extracted from the query is exactly the same as an entity mentioned in the paragraph."
        "Only consider the entity as the same if the full name exactly match. Partial matches, such as different middle names or additional qualifiers, should be considered as not the same."
        "Secondly, if the entities are exactly the same, determine if the paragraph contains any description that match the given relationship."
        "If there is a description that match the given relationship, extract the corresponding sentence(s) from the paragraph."
        "If there is no matched description, simply state 'No relevant description found'.\n"
        "Finally, explain the two matching judgments you made.\n\n"
        "Provide the output in the following format:\n"
        "Subject Match: true/false\n"
        "Relationship Match: true/false\n"
        "Description: XXX\n"
        "Explanation: XXX\n\n"},
        {"role": "user", "content": "### Subject: {subject}\n"
        "Relationship: {relationship}\n"
        "Paragraph: {paragraph}\n\n### Output: \n"}
    ], 
    "prompt_for_combine":[{"role": "system", "content":
        "You are provided with an instruction and some retrieved texts (wrapped in <paragraph></paragraph> tags). "
        "Your task is to select credible texts to answer the instruction. "
        "You should categorize the texts and reason out if they refer to the same entity based on the content. "
        "Integrate the texts referring to the same entity to provide a comprehensive answer." 
        "If the texts refer to entities that are the same name but different, answer in separate paragraphs."
        "Provide the detailed answer based on the paragraphs in the following format:\n"
        "Answer: XXX\n\n"},
        {"role": "user", "content":
        "### Instruction: {instruction}\n"
        "### Retrieved Texts:\n{paragraphs}\n\n"
        "### Answer:"}
    ],
    "prompt_for_combine_no_retrieval":[{"role": "system", "content":
        "You are provided with an instruction from the instruction. "
        "Your task is to answer the instruction. "
        "Provide the detailed answer in the following format:\n"
        "Answer: XXX\n\n"},
        {"role": "user", "content":
        "### Instruction: {instruction}\n"
        "### Answer:"}
    ],
    "prompt_for_splitting": [
    {
      "role": "system",
      "content": 
        "You will receive an instruction.The instruction may contain one or more entities, and additional knowledge about those entities needs to be retrieved in order to answer the query."
        "Analyze the instruction to determine if it requires multiple retrievals based on the number of entities or pieces of knowledge needed."
        "If the instruction requires multiple retrievals, split the instruction into sub-questions."
        "Each sub-question should target a specific entity or piece of knowledge needed to answer the main question."
        "If the instruction contains only one entity to be retrieved, then the sub-question count is 1 and the content of the sub-question should be exactly the same as the original instruction."
        "Note that count is the same as the number of entities to be retrieved in instruction and is a positive integer with a minimum value of 1."
        "Please give an explanation of your entire reasoning process."
        "Output the explanation of the instruction, the number of sub-questions and their content in the following format:\n"
        "Explanation: {XXX}"
        "Sub-question Count: {count}\n"
        "Sub-questions:\n"
        "1. {sub-question-1}\n"
        "2. {sub-question-2}\n"
        "... (and so on)"
        
    },
    {
      "role": "user",
      "content": "### Instruction: \n{instruction}\n"
    }
  ]

    
}

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
             "med": "Given four answer candidates, A, B, C and D, choose the best answer choice."}

rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]",
                          "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                        "[Utility:3]", "[Utility:4]", "[Utility:5]"]
ground_tokens_names = ["[Fully supported]",
                       "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text


def postprocess(pred):
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                      "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")

    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def preprocess_input(input_data, task):
    if task == "factscore":
        for item in input_data:
            item["instruction"] = item["input"]
            item["output"] = [item["output"]
                              ] if "output" in item else [item["topic"]]
        return input_data

    elif task == "qa":
        for item in input_data:
            if "instruction" not in item:
                item["instruction"] = item["question"]
            if "answers" not in item and "output" in item:
                item["answers"] = "output"
        return input_data

    elif task in ["asqa", "eli5"]:
        processed_input_data = []
        for instance_idx, item in enumerate(input_data["data"]):
            prompt = item["question"]
            instructions = TASK_INST[task]
            prompt = instructions + "## Input:\n\n" + prompt
            entry = copy.deepcopy(item)
            entry["instruction"] = prompt
            processed_input_data.append(entry)
        return processed_input_data


def postprocess_output(input_instance, prediction, task, intermediate_results=None):
    if task == "factscore":
        return {"input": input_instance["input"], "output": prediction, "topic": input_instance["topic"], "cat": input_instance["cat"]}

    elif task == "qa":
        input_instance["pred"] = prediction
        return input_instance

    elif task in ["asqa", "eli5"]:
        # ALCE datasets require additional postprocessing to compute citation accuracy.
        final_output = ""
        docs = []
        if "splitted_sentences" not in intermediate_results:
            input_instance["output"] = postprocess(prediction)

        else:
            for idx, (sent, doc) in enumerate(zip(intermediate_results["splitted_sentences"][0], intermediate_results["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = postprocess(sent)
                final_output += postprocessed_result[:-
                                                     1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if final_output[-1] == " ":
                final_output = final_output[:-1]
            input_instance["output"] = final_output
        input_instance["docs"] = docs
        return input_instance

def process_arc_instruction(item, instruction):
    choices = item["choices"]
    answer_labels = {}
    for i in range(len(choices["label"])):
        answer_key = choices["label"][i]
        text = choices["text"][i]
        if answer_key == "1":
            answer_labels["A"] = text
        if answer_key == "2":
            answer_labels["B"] = text
        if answer_key == "3":
            answer_labels["C"] = text
        if answer_key == "4":
            answer_labels["D"] = text
        if answer_key in ["A", "B", "C", "D"]:
            answer_labels[answer_key] = text

    if "D" not in answer_labels:
        answer_labels["D"] = ""
    choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
    if "E" in answer_labels:
        choices += "\nE: {}".format(answer_labels["E"])
    processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
    return processed_instruction


def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output
