from .. import GPT2Tokenizer

__all__ = ['BartTokenizer']


class BartTokenizer(GPT2Tokenizer):
    r"""
    Construct a BART tokenizer.

    :class:`~transformers.BartTokenizer` is identical to :class:`~transformers.RobertaTokenizer`. Refer to superclass
    :class:`~transformers.GPT2Tokenizer` for usage examples and documentation concerning the initialization
    parameters and other methods.
    """
    # merges and vocab same as GPT2
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "bart-base": "",
        },
        "merges_file": {
            "bart-base": "",
        }
    }
    pretrained_init_configuration = {"bart-base": {"do_lower_case": True}, }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
