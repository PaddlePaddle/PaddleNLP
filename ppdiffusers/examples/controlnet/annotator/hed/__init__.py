import torch
import paddle_patch

import numpy as np
import cv2
import os
import paddle
from einops import rearrange
from annotator.util import annotator_ckpts_path

from cldm.model import load
from paddlenlp.utils import load_torch

class Network(paddle.nn.Layer):
    def __init__(self, model_path):
        super().__init__()

        self.netVggOne = paddle.nn.Sequential(
            paddle.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU()
        )

        self.netVggTwo = paddle.nn.Sequential(
            paddle.nn.MaxPool2d(kernel_size=2, stride=2),
            paddle.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU()
        )

        self.netVggThr = paddle.nn.Sequential(
            paddle.nn.MaxPool2d(kernel_size=2, stride=2),
            paddle.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU()
        )

        self.netVggFou = paddle.nn.Sequential(
            paddle.nn.MaxPool2d(kernel_size=2, stride=2),
            paddle.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU()
        )

        self.netVggFiv = paddle.nn.Sequential(
            paddle.nn.MaxPool2d(kernel_size=2, stride=2),
            paddle.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            paddle.nn.ReLU()
        )

        self.netScoreOne = paddle.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = paddle.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = paddle.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = paddle.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = paddle.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = paddle.nn.Sequential(
            paddle.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            paddle.nn.Sigmoid()
        )

        from collections import OrderedDict
        def convert(model_data):
            new_model_data = OrderedDict()
            for k, v in model_data.items():
                print(k)
                new_model_data[k.replace('module', 'net')] = v.detach().numpy()
            return new_model_data

        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in load(model_path).items()})
        # self.load_state_dict(convert(torch.load(model_path)))
        # self.load_state_dict(convert(torch.load(model_path)))

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - paddle.to_tensor([104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype,).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = paddle.nn.functional.interpolate(tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = paddle.nn.functional.interpolate(tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = paddle.nn.functional.interpolate(tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = paddle.nn.functional.interpolate(tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = paddle.nn.functional.interpolate(tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(paddle.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))

class HEDdetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth"
        modelpath = os.path.join(annotator_ckpts_path, "network-bsds500.pth")
        if not os.path.exists(modelpath):
            from paddlenlp.utils.downloader import get_path_from_url_with_filelock
            get_path_from_url_with_filelock(remote_model_path, root_dir=annotator_ckpts_path)
        self.netNetwork = Network(modelpath).cuda().eval()

    def __call__(self, input_image):
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with paddle.no_grad():
            image_hed = paddle.to_tensor(input_image).float().cuda()
            image_hed = image_hed / 255.0
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_hed)[0]
            edge = (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return edge[0]


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z
