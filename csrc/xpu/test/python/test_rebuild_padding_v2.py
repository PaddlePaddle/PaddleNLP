# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import unittest
from paddlenlp_ops import rebuild_padding_v2

np.random.seed(2024)

class GetRebuildPaddingV2Test(unittest.TestCase):
    def test_rebuild_padding_v2(self):
      max_len = 10
      seq_lens = np.array([4, 3, 6], "int32").reshape(-1, 1)
      seq_lens_decoder = np.zeros_like(seq_lens).astype("int32")

      cum_offsets = np.insert(np.cumsum((max_len - seq_lens).flatten(), -1, "int32"),0,0)[:-1]
      token_num = np.sum(seq_lens)
      bs = seq_lens.shape[0]
      dim_emb = 129
      tmp_out = np.random.random((token_num, dim_emb)).astype("float16")
      # print("tmp_out:\n", paddle.to_tensor(tmp_out))
      # print("cum_offsets:\n", paddle.to_tensor(cum_offsets))
      # print("seq_lens_decoder:\n", paddle.to_tensor(seq_lens_decoder))
      # print("seq_lens:\n", paddle.to_tensor(seq_lens))


      out = rebuild_padding_v2(
          paddle.to_tensor(tmp_out),
          paddle.to_tensor(cum_offsets),
          paddle.to_tensor(seq_lens_decoder),
          paddle.to_tensor(seq_lens),
          max_len
      )

      def rebuild_padding_cpu(tmp_out, cum_offsets, seq_lens_decoder, seq_len_encoder, max_len):
        bs = seq_lens.shape[0]
        dim_emb = tmp_out.shape[1]
        output_data = np.zeros((bs, dim_emb)).flatten()
        seq_len = max_len
        tmp_out = tmp_out.flatten()
        for i in range(bs*dim_emb):
          bi = i // dim_emb
          bias_idx = i % dim_emb
          seq_id = 0
          # just encoder or stop, get last token; just decoder, get first token.
          if (seq_lens_decoder[bi] == 0):
              if seq_len_encoder[bi] != 0:
                  seq_id = seq_len_encoder[bi] - 1
              else:
                  continue
          ori_token_idx = bi * seq_len - cum_offsets[bi] + seq_id
          src_offset = ori_token_idx * dim_emb + bias_idx
          output_data[i] = tmp_out[src_offset]
        return output_data.reshape(bs, dim_emb)

      out_ = rebuild_padding_cpu(tmp_out, cum_offsets, seq_lens_decoder, seq_lens, max_len)

      np.testing.assert_allclose(out.numpy(), out_, atol=1e-05, rtol=1e-05)

if __name__ == '__main__':
    unittest.main()