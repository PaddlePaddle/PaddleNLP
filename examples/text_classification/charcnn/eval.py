import os
import argparse
import datetime
import sys
import errno
from tqdm import tqdm

import paddle
from paddle.io import DataLoader
import paddle.nn.functional as F

from model.char_cnn import CharCNN
from model.data_loader import AGNEWs
from model.metric import print_f_score

parser = argparse.ArgumentParser(description='Character level CNN text classifier testing',
                                 formatter_class=argparse.RawTextHelpFormatter)
# model
parser.add_argument('--model-path', default=None,
                    help='Path to pre-trained acouctics model created by DeepSpeech training')
parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('--l0', type=int, default=1014, help='maximum length of input sequence to CNNs [default: 1014]')
parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('--is_small', type=bool, default=False, help='use small CharCNN model')

# data
parser.add_argument('--test-path', metavar='DIR',
                    help='path to testing data csv', default='data/ag_news_csv/test.csv')
parser.add_argument('--batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('--alphabet-path', default='config/alphabet.json', help='Contains all characters for prediction')

# device
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
parser.add_argument('--device', type=str, default='gpu:0')

# logging options
parser.add_argument('--save-folder', default='Results/', help='Location to save epoch models')
args = parser.parse_args()


if __name__ == '__main__':
    paddle.set_device(args.device)

    # load testing data
    print("\nLoading testing data...")
    test_dataset = AGNEWs(label_data_path=args.test_path, alphabet_path=args.alphabet_path)
    print("Transferring testing data to iterator...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    _, num_class_test = test_dataset.getClassWeight()
    print('\nNumber of testing samples: ' + str(test_dataset.__len__()))
    for i, c in enumerate(num_class_test):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))

    args.num_features = len(test_dataset.alphabet)
    model = CharCNN(args.num_features, len(num_class_test), args.dropout, args.is_small)
    print("=> loading weights from '{}'".format(args.model_path))
    assert os.path.isfile(args.model_path), "=> no checkpoint found at '{}'".format(args.model_path)
    checkpoint = paddle.load(args.model_path)
    model.set_state_dict(checkpoint['state_dict'])

    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    print('\nTesting...')
    for i_batch, (data) in enumerate(tqdm(test_loader)):
        inputs, target = data
        inputs = paddle.to_tensor(inputs)
        target = paddle.to_tensor(target)
        size += len(target)

        logit = model(inputs)
        predicates = paddle.argmax(logit, 1)
        accumulated_loss += F.nll_loss(logit, target).numpy()[0]
        # print(type(target.data))
        corrects += paddle.to_tensor((paddle.argmax(logit, 1) == target), dtype='int64').sum().numpy()[0]
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += target.cpu().numpy().tolist()

    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    print('\rEvaluation - loss: {:.6f}  acc: {:.2f}%({}/{}) error: {:.2f}'.format(avg_loss,
                                                                     accuracy,
                                                                     corrects,
                                                                     size,
                                                                     100 - accuracy))
    print_f_score(predicates_all, target_all)
