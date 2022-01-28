import logging
import argparse
import pickle
from paddle.io import Dataset, DataLoader
import numpy as np
from utils.reading_comprehension.squad_get_predictions import *
from utils.reading_comprehension.squad_postprocess import *
from paddlenlp.transformers import LukeTokenizer
from paddlenlp.transformers import LukePretrainedModel
import paddle.nn as nn
from paddle.nn import CrossEntropyLoss
from utils.trainer import *

parser = argparse.ArgumentParser(description="LUKE FOR MRC")

parser.add_argument("--output_dir",
                    type=str,
                    required=True)
parser.add_argument("--data_dir",
                    type=str,
                    required=True)
parser.add_argument("--do_eval",
                    type=bool,
                    default=True)
parser.add_argument("--do_train",
                    type=bool,
                    default=True)
parser.add_argument("--eval_batch_size",
                    type=int,
                    default=32)
parser.add_argument("--num_train_epochs",
                    type=int,
                    default=2)
parser.add_argument("--seed",
                    type=int,
                    default=42)
parser.add_argument("--train_batch_size",
                    type=int,
                    default=8)
parser.add_argument("--device",
                    type=str,
                    default='gpu')
parser.add_argument("--gradient_accumulation_steps",
                    type=int,
                    default=3)
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.01)
parser.add_argument("--warmup_proportion",
                    type=float,
                    default=0.06)
parser.add_argument("--learning_rate",
                    type=float,
                    default=20e-6)
parser.add_argument("--adam_b1",
                    type=float,
                    default=0.9)
parser.add_argument("--adam_b2",
                    type=float,
                    default=0.99)
parser.add_argument("--model_type",
                    type=str,
                    default='luke-base'
                    )

args = parser.parse_args()
args.tokenizer = LukeTokenizer.from_pretrained(args.model_type)


class LukeForReadingComprehension(LukePretrainedModel):
    def __init__(self, luke):
        super(LukeForReadingComprehension, self).__init__()
        self.luke = luke  # luke模型用于特征抽取
        self.qa_outputs = nn.Linear(self.luke.config['hidden_size'], 2)  # 预测器
        self.apply(self.init_weights)

    def forward(
            self,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            start_positions=None,
            end_positions=None,
    ):
        encoder_outputs = self.luke(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask)

        word_hidden_states = encoder_outputs[0][:, : word_ids.shape[1], :]
        logits = self.qa_outputs(word_hidden_states)
        start_logits, end_logits = paddle.split(logits, 2, -1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,)
        else:
            outputs = tuple()

        return outputs + (start_logits, end_logits,)


def to_numpy(value):
    return np.array(value, np.int64)


class DataGenerator(Dataset):
    def __init__(self, features, args):
        super(DataGenerator, self).__init__()
        self.args = args
        self.all_word_ids = [f['word_ids'] for f in features]
        self.all_word_segment_ids = [f['word_segment_ids'] for f in features]
        self.all_word_attention_mask = [f['word_attention_mask'] for f in features]
        self.all_entity_ids = [f['entity_ids'] for f in features]
        self.all_entity_position_ids = [f['entity_position_ids'] for f in features]
        self.all_entity_segment_ids = [f['entity_segment_ids'] for f in features]
        self.all_entity_attention_mask = [f['entity_attention_mask'] for f in features]
        self.all_start_positions = [f['start_positions'] for f in features] if not args.evaluate else None
        self.all_end_positions = [f['end_positions'] for f in features] if not args.evaluate else None
        self.all_example_index = [f['example_indices'] for f in features] if args.evaluate else None

    def __getitem__(self, item):
        word_ids = to_numpy(self.all_word_ids[item])
        word_segment_ids = to_numpy(self.all_word_segment_ids[item])
        word_attention_mask = to_numpy(self.all_word_attention_mask[item])
        entity_ids = to_numpy(self.all_entity_ids[item])
        entity_position_ids = to_numpy(self.all_entity_position_ids[item])
        entity_segment_ids = to_numpy(self.all_entity_segment_ids[item])
        entity_attention_mask = to_numpy(self.all_entity_attention_mask[item])
        start_positions = np.array(self.all_start_positions[item], dtype=np.int64) if not self.args.evaluate else 0
        end_positions = np.array(self.all_end_positions[item], dtype=np.int64) if not self.args.evaluate else 0
        example_index = self.all_example_index[item] if self.args.evaluate else 0
        return word_ids, \
               word_segment_ids, \
               word_attention_mask, \
               entity_ids, \
               entity_position_ids, \
               entity_segment_ids, \
               entity_attention_mask, \
               start_positions, \
               end_positions, \
               example_index

    def __len__(self):
        return len(self.all_word_ids)


def load_examples(args, evaluate=False):
    args.evaluate = evaluate
    features = []
    if not evaluate:
        logging.info('Loading the preprocess data......')
        data_file = args.data_dir + 'train.json'
    else:
        logging.info('Loading the preprocess data......')
        data_file = args.data_dir + 'eval_data.json'
    with open(data_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            features.append(json.loads(line))
            line = f.readline()

    if evaluate:
        data_generator = DataGenerator(features, args)
        dataloader = DataLoader(data_generator, batch_size=args.eval_batch_size)
        with open(args.data_dir + 'eval_obj.pickle', 'rb') as f:
            eval_obj = pickle.load(f)
        examples, features, processor = eval_obj.examples, eval_obj.features, eval_obj.processor
    else:
        data_generator = DataGenerator(features, args)
        dataloader = DataLoader(data_generator, batch_size=args.train_batch_size, shuffle=True)
        examples, features, processor = None, None, None

    return dataloader, examples, features, processor


@paddle.no_grad()
def evaluate(args, model):
    dataloader, examples, features, processor = load_examples(args, evaluate=True)
    all_results = []
    logging.info("evaluating the model......")
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
    for batch in tqdm(dataloader, desc="eval"):
        model.eval()
        outputs = model(word_ids=batch[0],
                        word_segment_ids=batch[1],
                        word_attention_mask=batch[2],
                        entity_ids=batch[3],
                        entity_position_ids=batch[4],
                        entity_segment_ids=batch[5],
                        entity_attention_mask=batch[6])

        for i, example_index in enumerate(batch[-1]):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            start_logits, end_logits = [o[i] for o in outputs]
            start_logits = start_logits.tolist()
            end_logits = end_logits.tolist()
            all_results.append(RawResult(unique_id, start_logits, end_logits))
    all_predictions = write_predictions(args, examples, features,
                                        all_results, 20, 30, False)
    SQuad_postprocess(os.path.join(args.data_dir, processor.dev_file),
                      all_predictions,
                      output_metrics="output.json")


if __name__ == '__main__':
    model = LukeForReadingComprehension.from_pretrained(args.model_type)
    train_dataloader, _, _, _ = load_examples(args, evaluate=False)
    num_train_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    trainer = Trainer(args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps)
    trainer.train()
    model.from_pretrained(args.output_dir)
    evaluate(args, model)
