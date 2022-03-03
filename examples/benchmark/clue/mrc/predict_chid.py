
import os
from tqdm import tqdm 
import argparse

import paddle
from paddle.io import TensorDataset,DataLoader
from paddlenlp.transformers import BertForMultipleChoice, BertTokenizer
from paddlenlp.transformers import ErnieForMultipleChoice,ErnieTokenizer
from CHID_preprocess import RawResult, get_final_predictions, write_predictions, generate_input, evaluate

def process_test_data(input_dir,tokenizer,max_seq_length,max_num_choices):

    # 测试文件有两个，一个是test1.0.json，另一个是test1.1.json。
    # 可以根据情况使用test的json文件，本项目默认使用的是test1.0.json
    predict_file='work/test1.0.json'

    test_example_file = os.path.join(input_dir, 'test_examples_{}.pkl'.format(str(max_seq_length)))
    test_feature_file = os.path.join(input_dir, 'test_features_{}.pkl'.format(str(max_seq_length)))

    test_features = generate_input(predict_file, None, test_example_file, test_feature_file, tokenizer,
                                   max_seq_length=max_seq_length, max_num_choices=max_num_choices,
                                   is_training=False)

    all_example_ids = [f.example_id for f in test_features]
    all_tags = [f.tag for f in test_features]
    all_input_ids = paddle.to_tensor([f.input_ids for f in test_features], dtype="int64")
    all_input_masks = paddle.to_tensor([f.input_masks for f in test_features], dtype="int64")
    all_segment_ids = paddle.to_tensor([f.segment_ids for f in test_features], dtype="int64")
    all_choice_masks = paddle.to_tensor([f.choice_masks for f in test_features], dtype="int64")
    all_example_index = paddle.arange(all_input_ids.shape[0], dtype="int64")

    test_data = TensorDataset([all_input_ids, all_input_masks, all_segment_ids, all_choice_masks,
                              all_example_index])

    return test_data,all_example_ids,all_tags,test_features


@paddle.no_grad()
def do_test(model, dev_data_loader,all_example_ids,all_tags,eval_features):

    all_results = []
    model.eval()
    output_dir='work'
    for step, batch in enumerate(tqdm(dev_data_loader)):

        input_ids, input_masks, segment_ids, choice_masks, example_indices=batch
        batch_logits = model(input_ids=input_ids, token_type_ids=segment_ids,attention_mask=input_masks)
        # loss = criterion(batch_logits, labels)

        # all_loss.append(loss.numpy())
        for i, example_index in enumerate(example_indices):
            logits = batch_logits[i].numpy().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                             example_id=all_example_ids[unique_id],
                                             tag=all_tags[unique_id],
                                             logit=logits))
                
    output_file = 'chid10_predict.json'
    print('decoder raw results')
    tmp_predict_file = os.path.join(output_dir, "test_raw_predictions.pkl")
    output_prediction_file = os.path.join(output_dir, output_file)
    results = get_final_predictions(all_results, tmp_predict_file, g=True)
    write_predictions(results, output_prediction_file)
    print('predictions saved to {}'.format(output_prediction_file))

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model_path",
        default="checkpoints",
        type=str,
        help="The  path of the checkpoints .",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    print(args.model_path)

    input_dir='output'
    max_num_choices=10
    MODEL_NAME = args.model_path
    max_seq_length=64
    batch_size=4

    tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)
    model = ErnieForMultipleChoice.from_pretrained(MODEL_NAME,
                                                num_choices=max_num_choices)
    test_data,all_example_ids,all_tags,test_features=process_test_data(input_dir,tokenizer,max_seq_length,max_num_choices)

    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    do_test(model, test_data_loader,all_example_ids,all_tags,test_features)