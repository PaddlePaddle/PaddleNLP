import argparse
import os
import os.path as osp
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddlenlp.data import Dict, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertTokenizer, VisualBertForVisualReasoning
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
        "--visual_feature_root",
        type=str,
        required=False,
        # default=None,
        default="../X_NLVR/data/detectron_fix_144",
        help="Train data path.")

parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default="test",
        help="Mode for testing data.")

parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=2,
        help="Batch images")

parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        default="visualbert-nlvr2",
        help="model_name_or_path")

parser.add_argument(
        "--num_classes",
        type=int,
        required=False,
        default=2,
        help="num_classes")

parser.add_argument(
        "--return_dict",
        type=bool,
        required=False,
        default=False,
        help="Model return type")

args = parser.parse_args()
# ===================================================================
        
def prepare_dev_features_single(example, tokenizer, args):
    data_root = args.visual_feature_root

    caption_a = example['caption_a']
    label = example['label']
    identifier = example['identifier']
    feature_path_0 = example['feature_path_0']
    feature_path_1 = example['feature_path_1']

    if "train" in identifier:
        folder = osp.join(data_root, "train/feature_1024dim")
    elif "dev" in identifier:
        folder = osp.join(data_root, "dev/feature_1024dim")
    elif "test1" in identifier:
        folder = osp.join(data_root, "test1/feature_1024dim")
    
    detectron_features_0 = np.load(os.path.join(folder, feature_path_0))
    detectron_features_1 = np.load(os.path.join(folder, feature_path_1))
    detectron_features = np.concatenate((detectron_features_0, detectron_features_1), axis = 0)
    visual_embeds = paddle.to_tensor(detectron_features)
    
    visual_embeddings_type_0 = np.zeros(detectron_features_0.shape[0])
    visual_embeddings_type_1 = np.ones(detectron_features_1.shape[0])
    visual_embeddings_type = np.concatenate((visual_embeddings_type_0, visual_embeddings_type_1), axis = 0)
    visual_token_type_ids = paddle.to_tensor(visual_embeddings_type, dtype=paddle.int64)
    
    visual_attention_mask = paddle.ones(visual_embeds.shape[:-1], dtype=paddle.int64)
    
    bert_feature = tokenizer.encode(caption_a, return_attention_mask=True)
    
    label = paddle.to_tensor(label, dtype=paddle.int64)
    
    data = {
        "input_ids": bert_feature["input_ids"],
        "token_type_ids": bert_feature["token_type_ids"],    
        "attention_mask": bert_feature["attention_mask"],    
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
        "labels": label
    }
    
    return data

# ===================================================================

dev_ds = load_dataset("nlvr2", splits=["dev"])
label_list = dev_ds.label_list
vocab_info = dev_ds.vocab_info

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dev_ds.map(
    partial(prepare_dev_features_single, tokenizer=tokenizer, args=args),
    batched=False,
    lazy=True, #!! To save GPU Memory
)

dev_batch_sampler = paddle.io.BatchSampler(
    dev_ds, batch_size=args.batch_size, shuffle=False
)

dev_batchify_fn = lambda samples, fn=Dict(
    {
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "attention_mask": Pad(axis=0, pad_val=0),
        "visual_embeds": Pad(axis=0),
        "visual_token_type_ids": Pad(axis=0),
        "visual_attention_mask": Pad(axis=0),
        "labels": Pad(axis=0),
    }
): fn(samples)

dev_data_loader = DataLoader(
    dataset=dev_ds,
    batch_sampler=dev_batch_sampler,
    collate_fn=dev_batchify_fn,
    num_workers=0,
    return_list=True,
)

# ===================================================================


model = VisualBertForVisualReasoning.from_pretrained(args.model_name_or_path, num_classes=args.num_classes)
model.eval()

all_logits = []
# ===================================================================


from paddlenlp.metrics import AccuracyAndF1

metric = AccuracyAndF1()

with paddle.no_grad():
    for batch_idx, batch in tqdm(enumerate(dev_data_loader), total=len(dev_data_loader)):
        input_ids, token_type_ids, attention_mask, visual_embeds, visual_token_type_ids, visual_attention_mask, labels = batch
        batch_input = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "labels": labels,
            "return_dict": args.return_dict
        }
        
        output = model(**batch_input)
        
        if not args.return_dict:
            loss = output[0]
            logits = output[1]
        else:
            loss = output['loss']
            logits = output['logits']
        
        correct = metric.compute(logits, labels)
        metric.update(correct)
        acc, precision, recall, f1, average_of_acc_and_f1 = metric.accumulate()

        for idx in range(logits.shape[0]):
            all_logits.append(logits.numpy()[idx])
        
        if batch_idx % 500 == 0:
            print("acc", acc)
            
print("Final acc", acc)
       