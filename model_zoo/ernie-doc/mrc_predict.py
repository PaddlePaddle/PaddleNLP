import argparse
import uuid
from functools import partial

import paddle

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieDocTokenizer
from collections import namedtuple
from paddlenlp.transformers import ErnieDocForQuestionAnswering

from data import MRCIterator
from metrics import compute_qa_predictions


# yapf: disable
# yapf: enable

def read_text(text):
    question, context = text.split("\001")
    yield {'question': question, 'context': context, "id": uuid.uuid4()}


def init_memory(batch_size, memory_length, d_model, n_layers):
    return [
        paddle.zeros([batch_size, memory_length, d_model], dtype="float32")
        for _ in range(n_layers)
    ]


@paddle.no_grad()
def predict(model, tokenizer, data_loader, memories):
    model.eval()
    all_results = []
    for step, batch in enumerate(data_loader, start=0):
        input_ids, position_ids, token_type_ids, attn_mask, start_position, \
        end_position, qids, gather_idx, need_cal_loss = batch

        start_logits, end_logits, memories = model(input_ids, memories,
                                                   token_type_ids, position_ids,
                                                   attn_mask)

        start_logits, end_logits, qids = list(
            map(lambda x: paddle.gather(x, gather_idx),
                [start_logits, end_logits, qids]))
        np_qids = qids.numpy()
        np_start_logits = start_logits.numpy()
        np_end_logits = end_logits.numpy()

        if int(need_cal_loss.numpy()) == 1:
            for idx in range(qids.shape[0]):
                qid_each = int(np_qids[idx])
                start_logits_each = [
                    float(x) for x in np_start_logits[idx].flat
                ]
                end_logits_each = [float(x) for x in np_end_logits[idx].flat]
                all_results.append(
                    RawResult(unique_id=qid_each,
                              start_logits=start_logits_each,
                              end_logits=end_logits_each))
    all_predictions_eval, all_nbest_eval = compute_qa_predictions(
        data_loader._batch_reader.examples, data_loader._batch_reader.features,
        all_results, 20, 100,
        True, tokenizer, 1)

    return all_predictions_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="The directory to  model.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="max length of sequence")
    parser.add_argument("--memory_length", default=128, type=int, help="max length of memory")
    parser.add_argument("--input_question", required=True, type=str, help="input text of question")
    parser.add_argument("--input_context", required=True, type=str, help="input text of context")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to train model, defaults to gpu.")

    args = parser.parse_args()
    # ya: enable
    paddle.set_device(args.device)
    txt = args.input_question + "\001" + args.input_context
    batch_size = args.batch_size
    memory_length = args.memory_length
    test_ds = load_dataset(read_text, text=txt, lazy=False)

    trainer_num = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    tokenizer = ErnieDocTokenizer.from_pretrained("ernie-doc-base-zh")
    model = ErnieDocForQuestionAnswering.from_pretrained(args.model_dir)
    print("load over!")

    model_config = model.ernie_doc.config

    create_memory = partial(init_memory, 1, memory_length,
                            model_config["hidden_size"],
                            model_config["num_hidden_layers"])
    memories = create_memory()
    test_ds_iter = MRCIterator(test_ds,
                               1,
                               tokenizer,
                               trainer_num,
                               trainer_id=rank,
                               memory_len=args.memory_length,
                               max_seq_length=args.max_seq_length,
                               mode="test",
                               random_seed=1)

    test_dataloader = paddle.io.DataLoader.from_generator(capacity=70,
                                                          return_list=True)
    test_dataloader.set_batch_generator(test_ds_iter, paddle.get_device())

    RawResult = namedtuple("RawResult",
                           ["unique_id", "start_logits", "end_logits"])

    result = predict(model, tokenizer, test_dataloader, memories)
    print(result)
