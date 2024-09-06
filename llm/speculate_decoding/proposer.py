from abc import ABC, abstractmethod
from paddlenlp_ops import speculate_update_seq_lens_this_time, ngram_match
import paddle


class Proposer(ABC):
    """
    Proposer 基类

    用于在投机解码框架中提供可扩展的draft tokens接口
    """

    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def run(self, model_inputs, **kargs):
        """
        run
        """
        raise NotImplementedError()


class AutogressiveProposer(Proposer):
    """
    用于自回归解码的Proposer

    没有draft token, 仅套用框架，将上一次自回归生成的token放在draft_tokens的第一个位置
    """

    def __init__(self, **kwargs):
        super().__init__()

    def run(self, model_inputs, **kargs):
        speculate_update_seq_lens_this_time(
            kargs["seq_lens_this_time"],
            model_inputs["seq_lens_encoder"],
            model_inputs["seq_lens_decoder"],
            kargs["real_batch_size"],
            1,
        )


class InferenceWithReferenceProposer(Proposer):
    """
    用于Inference with reference的Proposer

    在输入输出中匹配符合的tokens作为draft tokens
    """

    def __init__(self, max_draft_tokens, max_ngram_size, max_batch_size, **kwargs):
        super().__init__()
        self.max_ngram_size = max_ngram_size
        self.input_ids_len = paddle.zeros(shape=[max_batch_size, 1], dtype="int64").cpu()
        self.max_batch_size = max_batch_size
        self.max_draft_tokens = max_draft_tokens

    #TODO(Wanglongzhi2001): unnessary
    # def update(self, bid: int, seq_len: int):
    #     """
    #     update
    #     """
    #     self.input_ids_len[bid] = seq_len

    def run(self, model_inputs, **kargs):
        """
        run
        """
        draft_tokens = model_inputs["draft_tokens"].cpu()
        seq_lens_this_time = kargs["seq_lens_this_time"].cpu()
        seq_lens_encoder = model_inputs["seq_lens_encoder"].cpu()
        seq_lens_decoder = model_inputs["seq_lens_decoder"].cpu()
        ngram_match(
            model_inputs["input_ids_cpu"],
            self.input_ids_len.cpu(),
            model_inputs["pre_ids"].cpu(),
            model_inputs["step_idx"].cpu(),
            model_inputs["actual_draft_token_num"].cpu(),
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            kargs["real_batch_size"],
            self.max_ngram_size,
            self.max_draft_tokens,
        )
        model_inputs["draft_tokens"][:] = draft_tokens.cuda()
        model_inputs["seq_lens_encoder"][:] = seq_lens_encoder.cuda()
        kargs["seq_lens_this_time"][:] = seq_lens_this_time.cuda()