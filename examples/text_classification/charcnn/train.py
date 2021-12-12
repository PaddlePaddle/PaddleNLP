#!/usr/bin/env python3
import argparse
import errno
import os
import sys

import paddle
import paddle.nn.functional as F
from paddle import optimizer
from paddle.io import DataLoader

from model.metric import print_f_score
from model.data_loader import AGNEWs
from model.char_cnn import CharCNN
from utils.utils import set_seed

set_seed(42)

parser = argparse.ArgumentParser(description='Character level CNN text classifier training')
# data 
parser.add_argument('--train_path', metavar='DIR',
                    help='path to training data csv [default: data/ag_news_csv/train.csv]',
                    default='data/ag_news_csv/train.csv')
parser.add_argument('--val_path', metavar='DIR',
                    help='path to validation data csv [default: data/ag_news_csv/test.csv]',
                    default='data/ag_news_csv/test.csv')
parser.add_argument('--data_augment', type=bool, default=False, help='whether to use data augmentation')
parser.add_argument('--geo_aug', type=bool, default=False, help='use GeometricSynonymAug in paper')

# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.0003, help='initial learning rate [default: 0.0001]')
learn.add_argument('--epochs', type=int, default=100, help='number of epochs for train [default: 200]')
learn.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')
learn.add_argument('--grad_clip', default=5, type=int, help='Norm cutoff to prevent explosion of gradients')
learn.add_argument('--optimizer', default='AdamW',
                   help='Type of optimizer. SGD|Adam|AdamW are supported [default: Adam]')
learn.add_argument('--class_weight', default=None, action='store_true',
                   help='Weights should be a 1D Tensor assigning weight to each of the classes.')
learn.add_argument('--dynamic_lr', action='store_true', default=False, help='Use dynamic learning schedule.')
learn.add_argument('--milestones', nargs='+', type=int, default=[5, 10, 15],
                   help=' List of epoch indices. Must be increasing. Default:[5,10,15]')
learn.add_argument('--decay_factor', default=0.5, type=float,
                   help='Decay factor for reducing learning rate [default: 0.5]')

# model (text classifier)
cnn = parser.add_argument_group('Model options')
cnn.add_argument('--alphabet_path', default='config/alphabet.json', help='Contains all characters for prediction')
cnn.add_argument('--l0', type=int, default=1014, help='maximum length of input sequence to CNNs [default: 1014]')
cnn.add_argument('--shuffle', action='store_true', default=True, help='shuffle the data every epoch')
cnn.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
cnn.add_argument('--kernel_num', type=int, default=100, help='number of each kind of kernel')
cnn.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
cnn.add_argument('--is_small', type=bool, default=False, help='use small CharCNN model')

# device
device = parser.add_argument_group('Device options')
device.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
device.add_argument('--device', type=int, default=None)

# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='Turn on progress tracking per iteration for debugging')
experiment.add_argument('--init_from_ckpt', default='', help='Continue from checkpoint model')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true',
                        help='Enables checkpoint saving of model')
experiment.add_argument('--save_folder', default='output/models_AG_NEWS',
                        help='Location to save epoch models, training configurations and results.')
experiment.add_argument('--log_config', default=True, action='store_true', help='Store experiment configuration')
experiment.add_argument('--log_result', default=True, action='store_true', help='Store experiment result')
experiment.add_argument('--log_interval', type=int, default=100,
                        help='how many steps to wait before logging training status [default: 1]')
experiment.add_argument('--val_interval', type=int, default=600,
                        help='how many steps to wait before vaidation [default: 400]')
experiment.add_argument('--save_interval', type=int, default=5,
                        help='how many epochs to wait before saving [default:1]')


def train(train_loader, dev_loader, model, args):
    # dynamic learning scheme
    scheduler = args.lr
    if args.dynamic_lr and args.optimizer == 'SGD':
        scheduler = optimizer.lr.MultiStepDecay(learning_rate=args.lr, milestones=args.milestones,
                                                gamma=args.decay_factor)

    # clip gradient
    clip = paddle.nn.ClipGradByNorm(clip_norm=args.grad_clip)
    # optimization scheme
    if args.optimizer == 'Adam':
        optim = optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'SGD':
        optim = optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=0.9, grad_clip=clip)
    elif args.optimizer == 'AdamW':
        optim = optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)

    # continue training from checkpoint model
    if args.init_from_ckpt:
        print("=> loading checkpoint from '{}'".format(args.init_from_ckpt))
        assert os.path.isfile(args.init_from_ckpt), "=> no checkpoint found at '{}'".format(args.init_from_ckpt)
        checkpoint = paddle.load(args.init_from_ckpt)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint.get('iter', None)
        best_acc = checkpoint.get('best_acc', None)
        print("=> checkpoint best acc: {}".format(best_acc))
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 1
        else:
            start_iter += 1
        model.set_state_dict(checkpoint['state_dict'])
        optim.set_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 1
        start_iter = 1
        best_acc = None

    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        if args.dynamic_lr and args.optimizer != 'Adam':
            scheduler.step()
        _i_batch = 0
        for i_batch, data in enumerate(train_loader, start=start_iter):
            _i_batch = i_batch
            inputs, target = data
            inputs = paddle.to_tensor(inputs)
            target = paddle.to_tensor(target)

            logit = model(inputs)
            loss = F.nll_loss(logit, target)
            loss.backward()
            optim.step()
            optim.clear_grad()

            if args.verbose:
                print('\nTargets, Predicates')
                print(paddle.concat(
                    (target.unsqueeze(1), paddle.unsqueeze(paddle.argmax(logit, 1).reshape(target.shape), 1)), 1))
                print('\nLogit')
                print(logit)

            if i_batch % args.log_interval == 0:
                corrects = paddle.to_tensor((paddle.argmax(logit, 1) == target), dtype='int64').sum().numpy()[0]
                accuracy = 100.0 * corrects / args.batch_size
                print('Epoch[{}] Batch[{}] - loss: {:.5f}  lr: {:.5f}  acc: {:.2f}% {}/{}'.format(epoch,
                                                                                                  i_batch,
                                                                                                  loss.numpy()[0],
                                                                                                  optim._learning_rate,
                                                                                                  accuracy,
                                                                                                  corrects,
                                                                                                  args.batch_size,
                                                                                                  ))
                sys.stdout.flush()


        if args.checkpoint and epoch % args.save_interval == 0:
            file_path = '%s/CharCNN_epoch_%d.pth.tar' % (args.save_folder, epoch)
            print("\r=> saving checkpoint model to %s" % file_path)
            save_checkpoint(model, {'epoch': epoch,
                                    'optimizer': optim.state_dict(),
                                    'best_acc': best_acc},
                            file_path)

        # validation
        val_loss, val_acc = eval(dev_loader, model, epoch, _i_batch, optim, args)
        # save best validation epoch model
        if best_acc is None or val_acc > best_acc:
            file_path = '%s/CharCNN_best.pth.tar' % (args.save_folder)
            print("\r=> found better validated model, saving to %s" % file_path)
            save_checkpoint(model,
                            {'epoch': epoch,
                             'optimizer': optim.state_dict(),
                             'best_acc': best_acc},
                            file_path)
            best_acc = val_acc
        start_iter = 1
        print('\n')
        sys.stdout.flush()


def eval(data_loader, model, epoch_train, batch_train, optim, args):
    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    for i_batch, (data) in enumerate(data_loader):
        inputs, target = data

        size += len(target)
        target = target.squeeze()
        logit = model(inputs)
        predicates = paddle.argmax(logit, 1)
        accumulated_loss += F.nll_loss(logit, target).numpy()[0]
        corrects += paddle.to_tensor((paddle.argmax(logit, 1) == target), dtype='int64').sum().numpy()[0]
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += target.cpu().numpy().tolist()

    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.5f}  lr: {:.5f}  acc: {:.2f} ({}/{}) error: {:.2f}'.format(avg_loss,
                                                                                              optim._learning_rate,
                                                                                              accuracy,
                                                                                              corrects,
                                                                                              size,
                                                                                              100.0 - accuracy))
    print_f_score(predicates_all, target_all)
    print('\n')
    sys.stdout.flush()
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.5f},{:.2f},{:f}'.format(epoch_train,
                                                            batch_train,
                                                            avg_loss,
                                                            accuracy,
                                                            optim._learning_rate))

    return avg_loss, accuracy


def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    paddle.save(state, filename)


def make_data_loader(dataset_path, alphabet_path, l0, batch_size, num_workers, data_augment=False, geo_aug=False):
    print("\nLoading data from {}".format(dataset_path))
    dataset = AGNEWs(label_data_path=dataset_path, alphabet_path=alphabet_path, l0=l0, data_augment=data_augment, geo_aug=geo_aug)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
    return dataset, dataset_loader


def main():
    # parse arguments
    args = parser.parse_args()
    # gpu
    if args.cuda and args.device:
        paddle.set_device(f"gpu:{args.device}")

    # load train and dev data
    train_dataset, train_loader = make_data_loader(args.train_path,
                                                   args.alphabet_path, args.l0, args.batch_size,
                                                   args.num_workers, args.data_augment, args.geo_aug)
    dev_dataset, dev_loader = make_data_loader(args.val_path,
                                               args.alphabet_path, args.l0, args.batch_size, args.num_workers)

    # feature length
    args.num_features = len(train_dataset.alphabet)

    # get class weights
    class_weight, num_class_train = train_dataset.getClassWeight()
    _, num_class_dev = dev_dataset.getClassWeight()

    # when you have an unbalanced training set
    if args.class_weight != None:
        args.class_weight = paddle.to_tensor(class_weight, dtype='float32').sqrt_()
        # if args.cuda:
        #     args.class_weight = args.class_weight.cuda()

    print('\nNumber of training samples: {}'.format(str(train_dataset.__len__())))
    for i, c in enumerate(num_class_train):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))
    print('\nNumber of developing samples: {}'.format(str(dev_dataset.__len__())))
    for i, c in enumerate(num_class_dev):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))

    # make save folder
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    # args.save_folder = os.path.join(args.save_folder, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))

    # log result
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))
    # model
    model = CharCNN(args.num_features, len(num_class_train), args.dropout, is_small=args.is_small)
    print(model)

    # train 
    train(train_loader, dev_loader, model, args)


if __name__ == '__main__':
    main()
