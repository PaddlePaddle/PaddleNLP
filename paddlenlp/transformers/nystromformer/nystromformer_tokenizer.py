from paddlenlp.transformers import BertTokenizer


class NystromformerTokenizer(BertTokenizer):
    """
    Tokenizer used by pretrained Nystromformer.
    """

    resource_files_names = {"vocab_file": "vocab.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "nystromformer-512":
            # TODO: upload parameter file and get the link here.
            "https://to-be-uploaded"
        }
    }
    pretrained_init_configuration = {
        "nystromformer-base": {
            "do_lower_case": True
        },
    }
