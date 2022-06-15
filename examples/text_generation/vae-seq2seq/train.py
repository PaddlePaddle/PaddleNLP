#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle

from model import CrossEntropyWithKL, VAESeq2SeqModel, Perplexity, NegativeLogLoss, TrainCallback
from args import parse_args
from data import create_data_loader


def train(args):
    print(args)
    device = paddle.set_device(args.device)
    train_loader, dev_loader, test_loader, vocab, bos_id, pad_id, train_data_len = create_data_loader(
        args)

    net = VAESeq2SeqModel(embed_dim=args.embed_dim,
                          hidden_size=args.hidden_size,
                          latent_size=args.latent_size,
                          vocab_size=len(vocab) + 2,
                          num_layers=args.num_layers,
                          init_scale=args.init_scale,
                          enc_dropout=args.enc_dropout,
                          dec_dropout=args.dec_dropout)

    gloabl_norm_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)

    anneal_r = 1.0 / (args.warm_up * train_data_len / args.batch_size)
    cross_entropy = CrossEntropyWithKL(base_kl_weight=args.kl_start,
                                       anneal_r=anneal_r)
    model = paddle.Model(net)

    optimizer = paddle.optimizer.Adam(args.learning_rate,
                                      parameters=model.parameters(),
                                      grad_clip=gloabl_norm_clip)

    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    ppl_metric = Perplexity(loss=cross_entropy)
    nll_metric = NegativeLogLoss(loss=cross_entropy)

    model.prepare(optimizer=optimizer,
                  loss=cross_entropy,
                  metrics=[ppl_metric, nll_metric])

    model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=args.max_epoch,
              save_dir=args.model_path,
              shuffle=False,
              callbacks=[TrainCallback(ppl_metric, nll_metric, args.log_freq)],
              log_freq=args.log_freq)

    # Evaluation
    print('Start to evaluate on test dataset...')
    model.evaluate(test_loader, log_freq=len(test_loader))


if __name__ == '__main__':
    args = parse_args()
    train(args)
