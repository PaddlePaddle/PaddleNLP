#!/usr/bin/env python3
import paddle
import paddle.nn as nn


class CharCNN(paddle.nn.Layer):
    def __init__(self, num_features, num_classes, dropout, is_small=False):
        super(CharCNN, self).__init__()
        hidden_dim = 256 if is_small else 1024
        output_dim = 1024 if is_small else 2048
        self.conv1 = nn.Sequential(
            nn.Conv1D(
                num_features, hidden_dim, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1D(
                kernel_size=3, stride=3))
        self.conv2 = nn.Sequential(
            nn.Conv1D(
                hidden_dim, hidden_dim, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1D(
                kernel_size=3, stride=3))

        self.conv3 = nn.Sequential(
            nn.Conv1D(
                hidden_dim, hidden_dim, kernel_size=3, stride=1),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv1D(
                hidden_dim, hidden_dim, kernel_size=3, stride=1),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv1D(
                hidden_dim, hidden_dim, kernel_size=3, stride=1),
            nn.ReLU())

        self.conv6 = nn.Sequential(
            nn.Conv1D(
                hidden_dim, hidden_dim, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1D(
                kernel_size=3, stride=3))

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim * 34, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout))

        self.fc2 = nn.Sequential(
            nn.Linear(output_dim, output_dim), nn.ReLU(), nn.Dropout(p=dropout))

        self.fc3 = nn.Linear(output_dim, num_classes)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        # x (bs, alphabet_size: 70, l0: 1014)
        x = self.conv1(x)  # (bs, hidden_dim, 336)
        x = self.conv2(x)  # (bs, hidden_dim, 110)
        x = self.conv3(x)  # (bs, hidden_dim, 108)
        x = self.conv4(x)  # (bs, hidden_dim, 106)
        x = self.conv5(x)  # (bs, hidden_dim, 104)
        x = self.conv6(x)  # (bs, hidden_dim, 34)

        # collapse
        x = x.reshape([x.shape[0], -1])  # (bs, hidden_dim *
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.log_softmax(x)
        return x
