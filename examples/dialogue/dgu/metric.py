import numpy as np

import paddle
import paddle.nn as nn
from paddle.metric import Metric


class RecallAtK(Metric):
    """
    Recall@K is the fraction of relevant results among the retrieved Top K 
    results, using to evaluate the performance of Dialogue Response Selection. 

    Noted that this class manages the Recall@K score only for binary
    classification task.
    """

    def __init__(self, name='Recall@K', *args, **kwargs):
        super(RecallAtK, self).__init__(*args, **kwargs)
        self._name = name
        self.softmax = nn.Softmax()
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.num_sampls = 0
        self.p_at_1_in_10 = 0.0
        self.p_at_2_in_10 = 0.0
        self.p_at_5_in_10 = 0.0

    def get_p_at_n_in_m(self, data, n, m, idx):
        """
        calculate precision in recall n
        """
        pos_score = data[idx][0]
        curr = data[idx:idx + m]
        curr = sorted(curr, key=lambda x: x[0], reverse=True)
        if curr[n - 1][0] <= pos_score:
            return 1
        return 0

    def update(self, logits, labels):
        """
        Update the states based on the current mini-batch prediction results.

        Args:
            logits (Tensor): The predicted value is a Tensor with 
                shape [batch_size, 2] and type float32 or float64.
            labels (Tensor): The ground truth value is a 2D Tensor, 
                its shape is [batch_size, 1] and type is int64.
        """
        probs = self.softmax(logits)
        probs = probs.numpy()
        labels = labels.numpy()
        assert probs.shape[0] == labels.shape[0]
        data = []
        for prob, label in zip(probs, labels):
            data.append((prob[1], label))
        assert len(data) % 10 == 0

        length = int(len(data) / 10)
        self.num_sampls += length
        for i in range(length):
            idx = i * 10
            assert data[idx][1] == 1
            self.p_at_1_in_10 += self.get_p_at_n_in_m(data, 1, 10, idx)
            self.p_at_2_in_10 += self.get_p_at_n_in_m(data, 2, 10, idx)
            self.p_at_5_in_10 += self.get_p_at_n_in_m(data, 5, 10, idx)

    def accumulate(self):
        """
        Calculate the final Recall@K.

        Returns:
            A list with scaler float: results of the calculated R1@K, R2@K, R5@K.
        """
        metrics_out = [
            self.p_at_1_in_10 / self.num_sampls,
            self.p_at_2_in_10 / self.num_sampls,
            self.p_at_5_in_10 / self.num_sampls
        ]
        return metrics_out

    def name(self):
        """
        Returns metric name
        """
        return self._name


class JointAccuracy(Metric):
    """
    The joint accuracy rate is used to evaluate the performance of multi-turn
    Dialogue State Tracking. For each turn, if and only if all state in 
    state_list are correctly predicted, the dialog state prediction is 
    considered correct. And the joint accuracy rate is equal to 1, otherwise 
    it is equal to 0.
    """

    def __init__(self, name='JointAccuracy', *args, **kwargs):
        super(JointAccuracy, self).__init__(*args, **kwargs)
        self._name = name
        self.sigmoid = nn.Sigmoid()
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.num_samples = 0
        self.correct_joint = 0.0

    def update(self, logits, labels):
        """
        Update the states based on the current mini-batch prediction results.

        Args:
            logits (Tensor): The predicted value is a Tensor with 
                shape [batch_size,  num_classes] and type float32 or float64.
            labels (Tensor): The ground truth value is a 2D Tensor, 
                its shape is [batch_size, num_classes] and type is int64.
        """
        probs = self.sigmoid(logits)
        probs = probs.numpy()
        labels = labels.numpy()
        assert probs.shape[0] == labels.shape[0]
        assert probs.shape[1] == labels.shape[1]
        for i in range(probs.shape[0]):
            pred, refer = [], []
            for j in range(probs.shape[1]):
                if probs[i][j] >= 0.5:
                    pred.append(j)
                if labels[i][j] == 1:
                    refer.append(j)
            if not pred:
                pred = [np.argmax(probs[i])]
            if pred == refer:
                self.correct_joint += 1
        self.num_samples += probs.shape[0]

    def accumulate(self):
        """
        Calculate the final JointAccuracy.

        Returns:
            A scaler float: results of the calculated JointAccuracy.
        """
        joint_acc = self.correct_joint / self.num_samples
        return joint_acc

    def name(self):
        """
        Returns metric name
        """
        return self._name


class F1Score(Metric):
    """
    F1-score is the harmonic mean of precision and recall. Micro-averaging is 
    to create a global confusion matrix for all examples, and then calculate 
    the F1-score. This class is using to evaluate the performance of Dialogue 
    Slot Filling.
    """

    def __init__(self, name='F1Score', *args, **kwargs):
        super(F1Score, self).__init__(*args, **kwargs)
        self._name = name
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.tp = {}
        self.fn = {}
        self.fp = {}

    def update(self, logits, labels):
        """
        Update the states based on the current mini-batch prediction results.

        Args:
            logits (Tensor): The predicted value is a Tensor with 
                shape [batch_size, seq_len, num_classes] and type float32 or 
                float64.
            labels (Tensor): The ground truth value is a 2D Tensor, 
                its shape is [batch_size, seq_len] and type is int64.
        """
        probs = paddle.argmax(logits, axis=-1)
        probs = probs.numpy()
        labels = labels.numpy()
        assert probs.shape[0] == labels.shape[0]
        assert probs.shape[1] == labels.shape[1]
        for i in range(probs.shape[0]):
            start, end = 1, probs.shape[1]
            while end > start:
                if labels[i][end - 1] != 0:
                    break
                end -= 1
            prob, label = probs[i][start:end], labels[i][start:end]
            for y_pred, y in zip(prob, label):
                if y_pred == y:
                    self.tp[y] = self.tp.get(y, 0) + 1
                else:
                    self.fp[y_pred] = self.fp.get(y_pred, 0) + 1
                    self.fn[y] = self.fn.get(y, 0) + 1

    def accumulate(self):
        """
        Calculate the final micro F1 score.

        Returns:
            A scaler float: results of the calculated micro F1 score.
        """
        tp_total = sum(self.tp.values())
        fn_total = sum(self.fn.values())
        fp_total = sum(self.fp.values())
        p_total = float(tp_total) / (tp_total + fp_total)
        r_total = float(tp_total) / (tp_total + fn_total)
        if p_total + r_total == 0:
            return 0
        f1_micro = 2 * p_total * r_total / (p_total + r_total)
        return f1_micro

    def name(self):
        """
        Returns metric name
        """
        return self._name
