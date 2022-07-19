import numpy as np
import paddle
import config


def get_input_from_batch(batch):
    batch_size = len(batch.enc_lens)
    enc_batch = paddle.to_tensor(batch.enc_batch, dtype='int64')
    enc_padding_mask = paddle.to_tensor(batch.enc_padding_mask, dtype='float32')
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = paddle.to_tensor(batch.enc_batch_extend_vocab,
                                                  dtype='int64')
        # The variable max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = paddle.zeros((batch_size, batch.max_art_oovs))

    c_t_1 = paddle.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = paddle.zeros(enc_batch.shape)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage


def get_output_from_batch(batch):
    dec_batch = paddle.to_tensor(batch.dec_batch, dtype='int64')
    dec_padding_mask = paddle.to_tensor(batch.dec_padding_mask, dtype='float32')
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = paddle.to_tensor(dec_lens, dtype='float32')

    target_batch = paddle.to_tensor(batch.target_batch, dtype='int64')

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
