from templates import *

task_config = {
    "boolq": {
        "train_datasets": ["boolq"],
        "eval_datasets": [
            "boolq",
        ],
        "task_prompt_template": "%s\n",
        "trigger_tokens": "the correct answer is ",
        "generation_args": {
            # align with https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 32,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 32,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            },
        },
    },
    "commonsense": {
        "train_datasets": ["commonsense_170k"],
        "eval_datasets": [
            "boolq",
            "piqa",
            "social_i_qa",
            "hellaswag",
            "winogrande",
            "ARC-Easy",
            "ARC-Challenge",
            "openbookqa",
        ],
        "task_prompt_template": "%s\n",
        "trigger_tokens": "the correct answer is ",
        "generation_args": {
            # align with https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 32,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 32,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            },
        },
    },
    "math": {
        "train_datasets": ["math_10k"],
        "eval_datasets": [
            "MultiArith",
            "gsm8k",
            "SVAMP",
            "mawps",
            "AddSub",
            "AQuA",
            "SingleEq",
        ],
        "task_prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # slightly changed to optimize our performance on top of
            # https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 512,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            },
        },
    },
    "alpaca": {
        "train_datasets": ["alpaca_data_cleaned"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            },
        },
    },
    "instruct": {
        "train_datasets": ["instruct"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            },
        },
    },
    "ultrafeedback": {
        "train_datasets": ["ultrafeedback"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            },
        },
    },
    "ultrafeedback_pair": {
        "train_datasets": ["argilla/ultrafeedback-binarized-preferences-cleaned"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            },
        },
    },
    "glue": {
        "train_datasets": None,
        "eval_datasets": None,
        "task_prompt_template": None,
        "trigger_tokens": None,
    },
    "gsm8k": {
        "train_datasets": ["gsm8k"],
        "eval_datasets": ["gsm8k"],
        "task_prompt_template": gsm8k_template,
        "trigger_tokens": "First think step by step and then answer the final number.\n",
        "generation_args": {
            # default values are from LoftQ
            # https://arxiv.org/pdf/2310.08659.pdf
            True: {
                "max_new_tokens": 256,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 256,
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "do_sample": True,
            },
        },
    },
}
