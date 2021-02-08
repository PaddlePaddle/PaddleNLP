# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_input_descs(args):
    """
    Generate a dict mapping data fields to the corresponding data shapes and
    data types.
    """
    # The placeholder for batch_size in compile time. Must be -1 currently to be
    # consistent with some ops' infer-shape output in compile time, such as the
    # sequence_expand op used in beamsearch decoder.
    batch_size = None
    # The placeholder for squence length in compile time.
    seq_len = None
    # The head number.
    n_head = getattr(args, "n_head", 8)
    # The model dim.
    d_model = getattr(args, "d_model", 512)

    # Here list the data shapes and data types of all inputs.
    # The shapes here act as placeholder and are set to pass the infer-shape in
    # compile time.
    input_descs = {
        # The actual data shape of src_word is:
        # [batch_size, max_src_len_in_batch]
        "src_word": [(batch_size, seq_len), "int64", 2],
        # The actual data shape of src_pos is:
        # [batch_size, max_src_len_in_batch, 1]
        "src_pos": [(batch_size, seq_len), "int64"],
        # This input is used to remove attention weights on paddings in the
        # encoder.
        # The actual data shape of src_slf_attn_bias is:
        # [batch_size, n_head, max_src_len_in_batch, max_src_len_in_batch]
        "src_slf_attn_bias":
        [(batch_size, n_head, seq_len, seq_len), "float32"],
        # The actual data shape of trg_word is:
        # [batch_size, max_trg_len_in_batch, 1]
        "trg_word": [(batch_size, seq_len), "int64",
                     2],  # lod_level is only used in fast decoder.
        # The actual data shape of trg_pos is:
        # [batch_size, max_trg_len_in_batch, 1]
        "trg_pos": [(batch_size, seq_len), "int64"],
        # This input is used to remove attention weights on paddings and
        # subsequent words in the decoder.
        # The actual data shape of trg_slf_attn_bias is:
        # [batch_size, n_head, max_trg_len_in_batch, max_trg_len_in_batch]
        "trg_slf_attn_bias":
        [(batch_size, n_head, seq_len, seq_len), "float32"],
        # This input is used to remove attention weights on paddings of the source
        # input in the encoder-decoder attention.
        # The actual data shape of trg_src_attn_bias is:
        # [batch_size, n_head, max_trg_len_in_batch, max_src_len_in_batch]
        "trg_src_attn_bias":
        [(batch_size, n_head, seq_len, seq_len), "float32"],
        # This input is used in independent decoder program for inference.
        # The actual data shape of enc_output is:
        # [batch_size, max_src_len_in_batch, d_model]
        "enc_output": [(batch_size, seq_len, d_model), "float32"],
        # The actual data shape of label_word is:
        # [batch_size * max_trg_len_in_batch, 1]
        "lbl_word": [(None, 1), "int64"],
        # This input is used to mask out the loss of paddding tokens.
        # The actual data shape of label_weight is:
        # [batch_size * max_trg_len_in_batch, 1]
        "lbl_weight": [(None, 1), "float32"],
        # This input is used in beam-search decoder.
        "init_score": [(batch_size, 1), "float32", 2],
        # This input is used in beam-search decoder for the first gather
        # (cell states updation)
        "init_idx": [(batch_size, ), "int32"],
    }

    return input_descs


# Names of word embedding table which might be reused for weight sharing.
word_emb_param_names = (
    "src_word_emb_table",
    "trg_word_emb_table", )
# Names of position encoding table which will be initialized externally.
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table", )
# separated inputs for different usages.
encoder_data_input_fields = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias", )
decoder_data_input_fields = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "enc_output", )
label_data_input_fields = (
    "lbl_word",
    "lbl_weight", )
# In fast decoder, trg_pos (only containing the current time step) is generated
# by ops and trg_slf_attn_bias is not needed.
fast_decoder_data_input_fields = (
    "trg_word",
    "init_score",
    "init_idx",
    "trg_src_attn_bias", )
