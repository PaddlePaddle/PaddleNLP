import argparse
import copy
import json
import logging
import os
import os.path as osp
import random
import time
from collections import OrderedDict
from functools import partial

import numpy as np
import paddle
from paddlenlp.data import Dict, Pad
from paddlenlp.datasets.dataset import load_dataset
from paddlenlp.transformers import (BertTokenizer, VisualBertModel,
                                    LinearDecayWithWarmup,
                                    VisualBertForPreTraining)

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "visualbert": (VisualBertModel, VisualBertForPreTraining, BertTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="visualbert",
        type=str,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--bert_model_name",
        default="bert-base-uncased",
        type=str,
        help="Path to bert model or shortcut name selected in the list: " +
        ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--model_name_or_path",
        default="visualbert-vqa-coco-pre",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[0].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--image_feature_type",
        default="coco_detectron_fix_100",
        type=str,
        help="`coco_detectron_fix_100` for vqa-coco-pre; `coco_detectron_fix_144` for nlvr2-coco-pre; `coco_detectron_fix_100` for vqa-pre; `nlvr2_detectron_fix_144` for nlvr2-pre; "
    )
    parser.add_argument(
        "--dataset",
        default="coco_captions",
        type=str,
        help="coco_captions for coco pretrain")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="max length of each sequence")
    parser.add_argument(
        "--mask_prob",
        default=0.15,
        type=float,
        help="the probability of one word to be mask")
    parser.add_argument(
        "--train_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--eval_batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=10000,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--init_from_ckpt",
        action="store_true",
        help="Whether to load model checkpoint. if True, args.model_name_or_path must be dir store ckpt or will train from fresh start"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Whether to use float16(Automatic Mixed Precision) to train.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--eager_run", type=bool, default=True, help="Use dygraph mode.")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu"],
        help="The device to select to train the model, is must be cpu/gpu.")
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def perform_truncate(max_seq_length, text_a, text_b):
    if text_b is None:
        len_total = len(text_a) + 2
        text_a = text_a[:max_seq_length - 2]
    else:
        len_total = len(text_a) + len(text_b) + 3
        if len_total > max_seq_length:
            take_away_from_ctx = min((len_total - max_seq_length + 1) // 2,
                                     max(len(text_a) - 32, 0))
            take_away_from_answer = len_total - max_seq_length + take_away_from_ctx
            # Follows VCR, perform truncate from the front...
            text_a = text_a[take_away_from_ctx:]
            text_b = text_b[take_away_from_answer:]

    return text_a, text_b


def random_word(tokens, tokenizer, probability=0.15):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < probability:
            prob /= probability

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(
                    list(tokenizer.vocab.idx_to_token.items()))[1]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".
                    format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def prepare_train_features_single(example, tokenizer, is_two_sentence, args):

    feature = None

    image_feature_type = args.image_feature_type

    if image_feature_type == "coco_detectron_fix_100":
        image_id = example['image_id']
        split_name = example['split_name']
        image_file_name = "COCO_{}2014_{:0>12d}.npy".format(split_name,
                                                            image_id)
        if "train" in image_file_name:
            folder = osp.join(args.input_dir,
                              "data/detectron_fix_100/fc6/vqa/train2014")
        elif "val" in image_file_name:
            folder = osp.join(args.input_dir,
                              "data/detectron_fix_100/fc6/vqa/val2014")
        image_feat_variable = np.load(osp.join(folder, image_file_name))
        visual_token_type_ids = np.zeros(
            image_feat_variable.shape[:-1], dtype=np.int64)
        visual_attention_mask = np.ones(
            image_feat_variable.shape[:-1], dtype=np.int64)
        visual_embeds = image_feat_variable.copy()
        visual_token_type_ids = visual_token_type_ids.copy()
        visual_attention_mask = visual_attention_mask.copy()

    elif image_feature_type == "coco_detectron_fix_144":
        image_id = example['image_id']
        split_name = example['split_name']
        image_file_name = "COCO_{}2014_{:0>12d}.npy".format(split_name,
                                                            image_id)
        if "train" in image_file_name:
            folder = osp.join(
                args.input_dir,
                "data/detectron_fix_144/nlvr2/train/feature_1024dim")
        elif "val" in image_file_name:
            folder = osp.join(
                args.input_dir,
                "data/detectron_fix_144/nlvr2/val/feature_1024dim")
        image_feat_variable = np.load(osp.join(folder, image_file_name))
        visual_token_type_ids = np.zeros(
            image_feat_variable.shape[:-1], dtype=np.int64)
        visual_attention_mask = np.ones(
            image_feat_variable.shape[:-1], dtype=np.int64)
        visual_embeds = image_feat_variable.copy()
        visual_token_type_ids = visual_token_type_ids.copy()
        visual_attention_mask = visual_attention_mask.copy()

    elif image_feature_type == "nlvr2_detectron_fix_144":
        caption_a = example['caption_a']
        label = example['label']
        identifier = example['identifier']
        feature_path_0 = example['feature_path_0']
        feature_path_1 = example['feature_path_1']
        if "train" in identifier:
            folder = osp.join(args.input_dir,
                              "data/detectron_fix_144/train/feature_1024dim")
        elif "dev" in identifier:
            folder = osp.join(args.input_dir,
                              "data/detectron_fix_144/dev/feature_1024dim")
        elif "test1" in identifier:
            folder = osp.join(args.input_dir,
                              "data/detectron_fix_144/test1/feature_1024dim")

        detectron_features_0 = np.load(os.path.join(folder, feature_path_0))
        detectron_features_1 = np.load(os.path.join(folder, feature_path_1))
        detectron_features = np.concatenate(
            (detectron_features_0, detectron_features_1), axis=0)

        visual_embeds = detectron_features.copy()

        visual_embeddings_type_0 = np.zeros(
            detectron_features_0.shape[0], dtype=np.int64)
        visual_embeddings_type_1 = np.ones(
            detectron_features_1.shape[0], dtype=np.int64)
        visual_embeddings_type = np.concatenate(
            (visual_embeddings_type_0, visual_embeddings_type_1),
            axis=0,
            dtype=np.int64)
        visual_token_type_ids = visual_embeddings_type.copy()
        visual_attention_mask = np.ones(
            visual_embeds.shape[:-1], dtype=np.int64)

        # example["visual_embeds"] = visual_embeds.copy()
        # example["visual_token_type_ids"] = visual_token_type_ids.copy()
        # example["visual_attention_mask"] = visual_attention_mask.copy()

    else:
        raise NotImplementedError("Please use image features from disk")

    feature = OrderedDict({
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask
    })
    visual_len = feature["visual_embeds"].shape[0]
    assert visual_len == 100 or visual_len == 144

    # Two sentence:
    # task1: masked language modeling
    # task2: are those sampled two sentence matched both matched ?

    # Single sentence:
    # task1: masked language modeling
    if is_two_sentence:
        tokens_a = tokenizer.tokenize(example['caption_a'])
        tokens_b = tokenizer.tokenize(example['caption_b'])
        tokens_a, tokens_b = perform_truncate(args.max_seq_length, tokens_a,
                                              tokens_b)
        raw_tokens_a, raw_tokens_b = tokens_a.copy(), tokens_b.copy()
        tokens_a, t1_label = random_word(tokens_a, tokenizer, args.mask_prob)
        tokens_b, t2_label = random_word(tokens_b, tokenizer, args.mask_prob)
        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        tokens = []
        raw_tokens = []
        token_type_ids = []
        raw_token_type_ids = []
        tokens.append("[CLS]")
        raw_tokens.append("[CLS]")
        token_type_ids.append(0)
        raw_token_type_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        for token in raw_tokens_a:
            raw_tokens.append(token)
            raw_token_type_ids.append(0)
        tokens.append("[SEP]")
        raw_tokens.append("[SEP]")
        token_type_ids.append(0)
        raw_token_type_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            token_type_ids.append(1)
        for token in raw_tokens_b:
            raw_tokens.append(token)
            raw_token_type_ids.append(1)
        tokens.append("[SEP]")
        raw_tokens.append("[SEP]")
        token_type_ids.append(1)
        raw_token_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        raw_input_ids = tokenizer.convert_tokens_to_ids(raw_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        sentence_image_labels = np.array(
            [1]) if example['is_correct'] == True else np.array([0])

        while len(lm_label_ids) < len(tokens) + visual_len:
            lm_label_ids.append(-1)

        feature.update({
            "tokens": tokens.copy(),
            "visual_embeds": visual_embeds.copy(),
            "raw_input_ids": raw_input_ids.copy(),
            "input_ids": input_ids.copy(),
            "token_type_ids": token_type_ids.copy(),
            "attention_mask": attention_mask.copy(),
            "labels": lm_label_ids.copy(),
            "sentence_image_labels": sentence_image_labels.copy()
        })

    else:
        if args.dataset == "vqa2":
            question_tokens = example['question_str']
            answers = example['answers']
            answer = np.random.choice(answers)
            tokens_a = tokenizer.tokenize(question_tokens) + tokenizer.tokenize(
                answer)

        elif args.dataset == "nlvr2":
            tokens_a = tokenizer.tokenize(example['caption_a'])

        else:
            NotImplementedError(
                "Unsupported dataset {} for single sentence language modeling".
                format(args.dataset))

        raw_tokens_a = tokens_a.copy()

        tokens_a, t1_label = random_word(tokens_a, tokenizer, args.mask_prob)
        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = ([-1] + t1_label + [-1])
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0

        tokens = []
        raw_tokens = []
        token_type_ids = []
        raw_token_type_ids = []
        tokens.append("[CLS]")
        raw_tokens.append("[CLS]")
        token_type_ids.append(0)
        raw_token_type_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        for token in raw_tokens_a:
            raw_tokens.append(token)
            raw_token_type_ids.append(0)
        tokens.append("[SEP]")
        raw_tokens.append("[SEP]")
        token_type_ids.append(0)
        raw_token_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        raw_input_ids = tokenizer.convert_tokens_to_ids(raw_tokens)

        while len(lm_label_ids) < len(tokens) + visual_len:
            lm_label_ids.append(-1)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        feature.update({
            "tokens": tokens.copy(),
            "visual_embeds": visual_embeds.copy(),
            "raw_input_ids": raw_input_ids.copy(),
            "input_ids": input_ids.copy(),
            "token_type_ids": token_type_ids.copy(),
            "attention_mask": attention_mask.copy(),
            "labels": lm_label_ids.copy()
        })

        if args.dataset == "nlvr2":
            feature.update({
                "visual_token_type_ids": visual_token_type_ids.copy(),
                "visual_attention_mask": visual_attention_mask.copy(),
            })

    return feature


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      use_gpu=True,
                      data_collator=None):
    """
    Creats dataloader.
    Args:
        dataset(obj:`paddle.io.Dataset`):
            Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`):
            If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): 
            The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`True`):
            Whether to use gpu to run.
    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=True)
        dataloader = paddle.io.DataLoader(
            dataset,
            batch_sampler=sampler,
            return_list=True,
            collate_fn=data_collator,
            num_workers=1)
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader = paddle.io.DataLoader(
            dataset,
            batch_sampler=sampler,
            return_list=True,
            collate_fn=data_collator,
            num_workers=1)

    return dataloader


def do_train(args):
    paddle.enable_static() if not args.eager_run else None
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())

    args.model_type = args.model_type.lower()
    base_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.bert_model_name)

    # Loads or initializes a model.
    pretrained_models = list(model_class.pretrained_init_configuration.keys())
    if args.model_name_or_path in pretrained_models:
        # model = model_class(
        #     base_class(**model_class.pretrained_init_configuration[
        #         args.model_name_or_path]))
        model = model_class.from_pretrained(args.model_name_or_path)
        args.init_from_ckpt = False
    else:
        if osp.isdir(args.model_name_or_path) and args.init_from_ckpt:
            # Load checkpoint
            with open(
                    osp.join(args.model_name_or_path, "run_states.json"),
                    'r') as f:
                config_dict = json.load(f)
                model_name = config_dict["model_name"]
            if model_name in pretrained_models:
                model = model_class.from_pretrained(model_name)
                model.set_state_dict(
                    paddle.load(
                        osp.join(args.model_name_or_path,
                                 "model_state.pdparams")))
            else:
                raise ValueError(
                    "initialize a model from ckpt need model_name "
                    "in model_config_file. The supported model_name "
                    "are as follows: {}".format(
                        tokenizer_class.pretrained_init_configuration.keys()))
        else:
            raise ValueError(
                "initialize a model need identifier or the "
                "directory of storing model. if use identifier, the supported model "
                "identifiers are as follows: {}, if use directory, "
                "make sure set init_from_ckpt as True".format(
                    model_class.pretrained_init_configuration.keys()))

    # criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # Loads dataset.
    tic_load_data = time.time()
    print("start load data : %s" %
          (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    is_two_sentence = False
    if args.dataset == "coco_captions":
        train_dataset = load_dataset(args.dataset, splits=["train"])
        is_two_sentence = True
    elif args.dataset == "vqa2":
        train_dataset = load_dataset(args.dataset, splits=["train"])
        is_two_sentence = False
    elif args.dataset == "nlvr2":
        train_dataset = load_dataset(args.dataset, splits=["train"])
        is_two_sentence = False
    else:

        raise NotImplementedError(
            "Only support `coco_captions` dataset for visualbert pretrain")

    print("load data done, total : %s s" % (time.time() - tic_load_data))

    # Reads data and generates mini-batches.
    if is_two_sentence:
        data_collator = lambda samples, fn=Dict({
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            "attention_mask": Pad(axis=0, pad_val=0),
            "visual_embeds": Pad(axis=0),
            "visual_token_type_ids": Pad(axis=0),
            "visual_attention_mask": Pad(axis=0, pad_val=1),
            "labels": Pad(axis=0, pad_val=-1),
            "sentence_image_labels": Pad(axis=0), }): fn(samples)
    else:
        data_collator = lambda samples, fn=Dict({
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            "attention_mask": Pad(axis=0, pad_val=0),
            "visual_embeds": Pad(axis=0),
            "visual_token_type_ids": Pad(axis=0),
            "visual_attention_mask": Pad(axis=0, pad_val=1),
            "labels": Pad(axis=0, pad_val=-1), }): fn(samples)

    trans_func = partial(
        prepare_train_features_single,
        tokenizer=tokenizer,
        is_two_sentence=is_two_sentence,
        args=args)
    train_dataset = train_dataset.map(trans_func, lazy=True)

    train_data_loader = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        mode='train',
        data_collator=data_collator)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_params)
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    print("start train : %s" %
          (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    trained_global_step = global_step = 0
    t_loss = paddle.to_tensor([0.0])
    log_loss = paddle.to_tensor([0.0])
    loss_list = []
    log_list = []
    tic_train = time.time()
    if osp.isdir(args.model_name_or_path) and args.init_from_ckpt:
        optimizer.set_state_dict(
            paddle.load(
                osp.join(args.model_name_or_path, "model_state.pdopt")))
        trained_global_step = global_step = config_dict["global_step"]
        if trained_global_step < num_training_steps:
            print(
                "[ start train from checkpoint ] we have already trained %s steps, seeking next step : %s"
                % (trained_global_step, trained_global_step + 1))
        else:
            print(
                "[ start train from checkpoint ] we have already trained %s steps, but total training steps is %s, please check configuration !"
                % (trained_global_step, num_training_steps))
            exit(0)

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            if trained_global_step > 0:
                trained_global_step -= 1
                continue
            global_step += 1
            return_dict = False
            if is_two_sentence:
                inputs = {
                    "input_ids": batch[0],
                    "token_type_ids": batch[1],
                    "attention_mask": batch[2],
                    "visual_embeds": batch[3],
                    "visual_token_type_ids": batch[4],
                    "visual_attention_mask": batch[5],
                    "labels": batch[6],
                    "sentence_image_labels": batch[7],
                    "return_dict": return_dict
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "token_type_ids": batch[1],
                    "attention_mask": batch[2],
                    "visual_embeds": batch[3],
                    "visual_token_type_ids": batch[4],
                    "visual_attention_mask": batch[5],
                    "labels": batch[6],
                    "return_dict": return_dict
                }

            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                outputs = model(**inputs)
                if not inputs['return_dict']:
                    loss = outputs[0]
                    # prediction_logits = outputs[1].cpu().detach().numpy()
                else:
                    loss = outputs['loss']
                    # prediction_logits = outputs['prediction_logits'].cpu().detach().numpy()
                    # seq_relationship_logits = outputs['seq_relationship_logits'].cpu().detach().numpy()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            optimizer.clear_grad()
            t_loss += loss.detach()

            if global_step % args.logging_steps == 0:
                local_loss = (t_loss - log_loss) / args.logging_steps
                if (paddle.distributed.get_world_size() > 1):
                    paddle.distributed.all_gather(loss_list, local_loss)
                    if paddle.distributed.get_rank() == 0:
                        log_str = (
                            "global step {0:d}/{1:d}, epoch: {2:d}, batch: {3:d}, "
                            "avg_loss: {4:.15f}, lr: {5:.10f}, speed: {6:.2f} s/it"
                        ).format(global_step, num_training_steps, epoch, step,
                                 float((paddle.stack(loss_list).sum() / len(
                                     loss_list)).numpy()),
                                 optimizer.get_lr(),
                                 (time.time() - tic_train) / args.logging_steps)
                        print(log_str)
                        log_list.append(log_str)
                        loss_list = []
                else:
                    log_str = (
                        "global step {0:d}/{1:d}, epoch: {2:d}, batch: {3:d}, "
                        "loss: {4:.15f}, lr: {5:.10f}, speed: {6:.2f} s/it"
                    ).format(global_step, num_training_steps, epoch, step,
                             float(local_loss.numpy()),
                             optimizer.get_lr(),
                             (time.time() - tic_train) / args.logging_steps)
                    print(log_str)
                    log_list.append(log_str)
                log_loss = paddle.to_tensor(t_loss)
                tic_train = time.time()
            if global_step % args.save_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    output_dir = osp.join(args.output_dir,
                                          "model_%d.pdparams" % global_step)
                    if not osp.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    config_to_save = copy.deepcopy(
                        model_to_save.visual_bert.config)
                    if 'self' in config_to_save:
                        del config_to_save['self']
                    run_states = {
                        "model_name": model_name
                        if args.init_from_ckpt else args.model_name_or_path,
                        "global_step": global_step,
                        "epoch": epoch,
                        "step": step,
                    }
                    with open(osp.join(output_dir, "model_config.json"),
                              'w') as f:
                        json.dump(config_to_save, f)
                    with open(osp.join(output_dir, "run_states.json"),
                              'w') as f:
                        json.dump(run_states, f)
                    paddle.save(model.state_dict(),
                                osp.join(output_dir, "model_state.pdparams"))
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(optimizer.state_dict(),
                                osp.join(output_dir, "model_state.pdopt"))
                    if len(log_list) > 0:
                        with open(osp.join(output_dir, "train.log"), 'w') as f:
                            for log in log_list:
                                if len(log.strip()) > 0:
                                    f.write(log.strip() + '\n')
            if global_step >= num_training_steps:
                del train_data_loader
                return


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    n_gpu = len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
    if args.device in "gpu" and n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=n_gpu)
    else:
        do_train(args)
