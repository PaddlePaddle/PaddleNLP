from paddlenlp.datasets.experimental import SQuAD
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict
from functools import partial
from paddle.io import DataLoader

train_ds, dev_ds = SQuAD().read_datasets('train_v2', 'dev_v2')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(len(train_ds))
print(len(dev_ds))
print(train_ds[0])

print('-----------------------------------------------------------')


def prepare_train_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    contexts = [examples[i]['context'] for i in range(5000)]
    questions = [examples[i]['question'] for i in range(5000)]

    tokenized_examples = tokenizer(
        questions, contexts, stride=128, max_seq_len=384)
    print(len(tokenized_examples))

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.

    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.

    # Let's label those examples!

    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        offsets = tokenized_example['offset_mapping']

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['segment_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']
        if i == 0:
            print(answer_starts)
            print(len(answer_starts))
        # If no answers are given, set the cls_index as answer.
        if len(answer_starts) == 0:
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
        else:
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 2
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[
                        token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples[i]["start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1

    return tokenized_examples


train_ds.map(prepare_train_features)

print(train_ds[0])
print(train_ds[1])
print(len(train_ds))
print('-----------------------------------------------------')

train_batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
    "segment_ids": Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
    "start_positions": Stack(dtype="int64"),  # start_pos
    "end_positions": Stack(dtype="int64")  # end_pos
}): fn(samples)

train_data_loader = DataLoader(
    dataset=train_ds,
    batch_size=8,
    collate_fn=train_batchify_fn,
    return_list=True)

for batch in train_data_loader:
    print(batch[0])
    print(batch[1])
    print(batch[2])
    print(batch[3])
    break
