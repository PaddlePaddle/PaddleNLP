#
#
#

from paddlenlp.transformers.bert.tokenizer import BertTokenizer

__all__ = ["ModernBertTokenizer"]


class ModernBertTokenizer(BertTokenizer):
    resource_files_names = {"vocab_file": "vocab.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "modernbert-base": "https://bj.bcebos.com/paddlenlp/models/transformers/modernbert/modernbert-base-vocab.txt",
            "modernbert-large": "https://bj.bcebos.com/paddlenlp/models/transformers/modernbert/modernbert-large-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "modernbert-base": {"do_lower_case": True},
        "modernbert-large": {"do_lower_case": True},
    }
