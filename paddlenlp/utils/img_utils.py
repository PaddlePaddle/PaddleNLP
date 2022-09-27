# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import copy
import uuid
import math
import json
import gzip
import tqdm
import random
import pickle
import re
import base64
from PIL import Image
from functools import cmp_to_key
import cv2
import numpy as np
from collections import Counter, defaultdict, namedtuple
from collections.abc import Sequence
from .log import logger


class BaseOperator(object):

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


class DecodeImage(BaseOperator):

    def __init__(self, to_rgb=True, with_mixup=False, with_cutmix=False):
        """ Transform the image data to numpy format.
        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
            with_cutmix (bool): whether or not to cutmix image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
        self.with_cutmix = with_cutmix
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if not isinstance(self.with_mixup, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            sample["image"] = base64.b64decode(
                sample["im_base64"].encode('utf-8'))

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode

        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            sample['w'] = im.shape[1]

        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array([im.shape[0], im.shape[1], 1.],
                                     dtype=np.float32)

        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'], context)

        # decode cutmix image
        if self.with_cutmix and 'cutmix' in sample:
            self.__call__(sample['cutmix'], context)

        # decode semantic label
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            sem_file = sample['semantic']
            sem = cv2.imread(sem_file, cv2.IMREAD_GRAYSCALE)
            sample['semantic'] = sem.astype('int32')

        return sample


class ResizeImage(BaseOperator):

    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True,
                 resize_box=False):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.
        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            max_size (int): the max size of image
            interp (int): the interpolation method
            use_cv2 (bool): use the cv2 interpolation method or use PIL
                interpolation method
            resize_box (bool): whether resize ground truth bbox annotations.
        """
        super(ResizeImage, self).__init__()
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        self.resize_box = resize_box
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}"
                .format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.max_size, int)
                and isinstance(self.interp, int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None, save_real_img=False):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]
            if 'im_info' in sample and sample['im_info'][2] != 1.:
                sample['im_info'] = np.append(list(sample['im_info']),
                                              im_info).astype(np.float32)
            else:
                sample['im_info'] = np.array(im_info).astype(np.float32)
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(im,
                            None,
                            None,
                            fx=im_scale_x,
                            fy=im_scale_y,
                            interpolation=self.interp)
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)

        if save_real_img:

            save_path = os.path.basename(sample["im_file"])
            save_path = os.path.join("./test/real_imgs", save_path)
            with open("./test/real_img_path.txt", "a") as f:
                f.write(save_path + "\n")
            cv2.imwrite(save_path, im)

        # add mask
        for key in sample.keys():
            if key.startswith("black_") and sample[key] is not None:
                for left, top, width, height in sample[key]:
                    left = int(left * im_scale_x)
                    top = int(top * im_scale_y)
                    width = int(width * im_scale_x)
                    height = int(height * im_scale_y)
                    im = cv2.rectangle(im, (left, top),
                                       (left + width, top + height), (0, 0, 0),
                                       thickness=-1)
        sample['image'] = im
        sample['scale_factor'] = [im_scale_x, im_scale_y] * 2
        if 'gt_bbox' in sample and self.resize_box and len(
                sample['gt_bbox']) > 0:
            bboxes = sample['gt_bbox'] * sample['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, resize_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, resize_h - 1)
            sample['gt_bbox'] = bboxes
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            semantic = sample['semantic']
            semantic = cv2.resize(semantic.astype('float32'),
                                  None,
                                  None,
                                  fx=im_scale_x,
                                  fy=im_scale_y,
                                  interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(gt_segm,
                           None,
                           None,
                           fx=im_scale_x,
                           fy=im_scale_y,
                           interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample


class Permute(BaseOperator):

    def __init__(self, to_bgr=True, channel_first=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        super(Permute, self).__init__()
        self.to_bgr = to_bgr
        self.channel_first = channel_first
        if not (isinstance(self.to_bgr, bool)
                and isinstance(self.channel_first, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            assert 'image' in sample, "image data not found"
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    if self.channel_first:
                        im = np.swapaxes(im, 1, 2)
                        im = np.swapaxes(im, 1, 0)
                    if self.to_bgr:
                        im = im[[2, 1, 0], :, :]
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


class NormalizeImage(BaseOperator):

    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[1, 1, 1],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list)
                and isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples


class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape
                              for data in samples]).max(axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros((im_c, max_shape[1], max_shape[2]),
                                  dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]
            if 'semantic' in data.keys() and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros((1, max_shape[1], max_shape[2]),
                                       dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data.keys() and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm

        return samples


class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            for i, (mask, downsample_ratio) in enumerate(
                    zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj,
                               gi] = np.log(gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj,
                               gi] = np.log(gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target
        return samples


def check(s):
    """Check whether is English"""
    my_re = re.compile(r'[A-Za-z0-9]', re.S)
    res = re.findall(my_re, s)
    if len(res):
        return True
    return False


def img2base64(img_path):
    """ get base64 """
    with open(img_path, "rb") as f:
        base64_str = base64.b64encode(f.read()).decode('utf-8')
    return base64_str


class Bbox(object):
    """
    The inner store format of `Bbox` is (left, top, width, height).

    The user may instance plenty of `Bbox`, thats why we insist the `Bbox` only contains four variables.
    """

    __slots__ = ["_c_left", "_c_top", "_c_width", "_c_height"]

    def __init__(self, left=0, top=0, width=0, height=0):
        """
        Constructor of `Bbox`.

        >> left: The most left position of bounding box.
        >> right: The most right position of bounding box.
        >> width: The width of bounding box.
        >> height: The height of bounding box.

        ^^ AssertionError: width and height must larger than 0.
        """
        assert width >= 0, "width {} must no less than 0".format(width)
        assert height >= 0, "height {} must no less than 0".format(height)
        self._c_left, self._c_top, self._c_width, self._c_height = left, top, width, height

    def __str__(self):
        """
        Reload the `str` operator.
        """
        return repr(self)

    def __repr__(self):
        """
        Reload the `repr` operator.
        """
        return "(x={}, y={}, w={}, h={})".format(self.left, self.top,
                                                 self.width, self.height)

    def __eq__(self, other):
        """
        if `self` is equal with given `other` box.

        >> other: The comparing box instance.

        << True if two box is equal else False.
        """
        return self.left == other.left and self.top == other.top \
               and self.width == other.width and self.height == other.height

    def tuple(self, precision=3):
        """
        Return the tuple format box.
        """
        return tuple(
            round(one, precision)
            for one in (self.left, self.top, self.width, self.height))

    def list_int(self):
        """
        Return the list(int) format box.
        """
        return list(
            int(one) for one in (self.left, self.top, self.width, self.height))

    def points_tuple(self, precision=3):
        """
        Return the coordinate of box
        """
        return tuple(
            round(one, precision)
            for one in (self.left, self.top, self.right, self.bottom))

    @property
    def left(self):
        """
        Visit the most left position of bounding box.
        """
        return self._c_left

    @left.setter
    def left(self, left):
        """
        Set the most left position of bounding box.
        """
        self._c_left = left

    @property
    def right(self):
        """
        Visit the most right position of bounding box.
        """
        return self._c_left + self._c_width

    @right.setter
    def right(self, right):
        """
        Set the most right position of bounding box.

        ^^ AssertionError: when right is less than left.
        """
        assert right >= self._c_left, "right {} < left {} is forbidden.".format(
            right, self._c_left)
        self._c_width = right - self._c_left

    @property
    def top(self):
        """
        Visit the most top position of bounding box.
        """
        return self._c_top

    @top.setter
    def top(self, top):
        """
        Set the most top position of bounding box.
        """
        self._c_top = top

    @property
    def bottom(self):
        """
        Visit the most bottom position of bounding box.
        """
        return self._c_top + self._c_height

    @bottom.setter
    def bottom(self, bottom):
        """
        Set the most bottom position of bounding box.

        ^^ AssertionError: when bottom is less than top.
        """
        assert bottom >= self._c_top, "top {} > bottom {} is forbidden.".format(
            self._c_top, bottom)
        self._c_height = bottom - self._c_top

    @property
    def width(self):
        """
        Visit the width of bounding box.
        """
        return self._c_width

    @width.setter
    def width(self, width):
        """
        Set the width of bounding box.

        ^^ AssertionError: when width is less than 0.
        """
        assert width >= 0, "width {} < 0 is forbidden.".format(width)
        self._c_width = width

    @property
    def height(self):
        """
        Visit the height of bounding box.
        """
        return self._c_height

    @height.setter
    def height(self, height):
        """
        Set the height of bounding box.

        ^^ AssertionError: when height is less than 0.
        """
        assert height >= 0, "height {} < 0 is forbidden.".format(height)
        self._c_height = height

    def is_cross_boundary(self, width, height, top=0, left=0):
        """
        If this box is cross boundary of given boundary. The boundary is start at (0, 0) by default.

        >> width: The width of boundary.
        >> height: The height of boundary.
        >> top: The top-left point location. Default at (0, 0)
        >> left: The top-left point location. Default at (0, 0)
        """
        boundary = Bbox(top, left, width, height)
        return boundary.contain(self)

    def is_vertical(self):
        """
        If this box is vertical.
        """
        return self.width < self.height

    def is_horizontal(self):
        """
        If this box is horizontal.
        """
        return self.width > self.height

    def is_square(self):
        """
        If this box is square.
        """
        return self.width == self.height

    def center(self):
        """
        Return the center point of this box.
        """
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def points(self):
        """
        Convert bounding box to main corner points (left, top) + (right, bottom).

        << Two tuple of points, left-top and right-bottom respectively.
        """
        return (self.left, self.top), (self.right, self.bottom)

    def contain(self, box):
        """
        If given `box` is contained by `self`.

        >> box: The box supposed to be contained.

        << True if `self` contains `box` else False
        """
        return self.left <= box.left and self.top <= box.top \
               and self.right >= box.right and self.bottom >= box.bottom

    def overlap_vertically(self, box):
        """
        If given `box` is overlap with `self` vertically.

        >> box: The comparing box.

        << True if overlap with each others vertically else False.
        """
        return not (self.top >= box.bottom or self.bottom <= box.top)

    def overlap_horizontally(self, box):
        """
        If given `box` is overlap with `self` horizontally.

        >> box: The comparing box.

        << True if overlap with each others horizontally else False.
        """
        return not (self.left >= box.right or self.right <= box.left)

    def overlap(self, box):
        """
        If given `box` is overlap with `self`.

        >> box: The comparing box.

        << True if overlap with each others else False.
        """
        return self.overlap_horizontally(box) and self.overlap_vertically(box)

    def hoverlap(self, box):
        """
        The value of overlapped horizontally.

        >> box: The calculating box.
        """
        if not self.overlap_horizontally(box):
            return 0

        return min(self.right, box.right) - max(self.left, box.left)

    def voverlap(self, box):
        """
        The value of overlap vertically.

        >> box: The calculating box.
        """
        if not self.overlap_vertically(box):
            return 0

        return min(self.bottom, box.bottom) - max(self.top, box.top)

    def hdistance(self, box):
        """
        The distance of two boxes horizontally.

        >> box: The calculating box.
        """
        if self.overlap_horizontally(box):
            return 0

        return max(self.left, box.left) - min(self.right, box.right)

    def vdistance(self, box):
        """
        The distance of two boxes vertically.

        >> box: The calculating box.
        """
        if self.overlap_vertically(box):
            return 0

        return max(self.top, box.top) - min(self.bottom, box.bottom)

    def area(self):
        """
        Calculate the area within the bounding box.
        """
        return self.width * self.height

    def translate(self, vector):
        """
        Translate box in the direction of vector
        """
        return Bbox(self.left + vector[0], self.top + vector[1], self.width,
                    self.height)

    @staticmethod
    def union(*boxes):
        """
        Calculate the union bounding box of given `boxes`.

        >> boxes: The boxes to calculate with.

        << The union `Bbox` of `boxes`.
        """
        left, top = min([box.left
                         for box in boxes]), min([box.top for box in boxes])
        right, bottom = max([box.right for box in boxes
                             ]), max([box.bottom for box in boxes])

        return Bbox.from_points((left, top), (right, bottom))

    @staticmethod
    def adjacency(boxa, boxb):
        """
        Calculate the adjacent bounding box of given boxes.

        >> boxa: The box to calculate with.
        >> boxb: The box to calculate with.

        << The adjacent `Bbox` of boxes.
        """
        horizon = [min(boxa.right, boxb.right), max(boxa.left, boxb.left)]
        vertical = [min(boxa.bottom, boxb.bottom), max(boxa.top, boxb.top)]

        left, right = min(horizon), max(horizon)
        top, bottom = min(vertical), max(vertical)

        return Bbox.from_points((left, top), (right, bottom))

    @staticmethod
    def intersection(*boxes):
        """
        Calculate the intersection bounding box of given `boxes`.

        >> boxes: The boxes to calculate with.

        << The intersection `Bbox` of `boxes`.
        """
        left, top = max(box.left for box in boxes), max(box.top
                                                        for box in boxes)
        right, bottom = min(box.right for box in boxes), min(box.bottom
                                                             for box in boxes)

        if left > right or top > bottom:
            return Bbox()

        return Bbox.from_points((left, top), (right, bottom))

    @staticmethod
    def iou(boxa, boxb):
        """
        Calculate the union area divided by intersection area.

        >> boxa: The box to calculate with.
        >> boxb: The box to calculate with.
        """
        return Bbox.intersection(boxa, boxb).area() / Bbox.union(boxa,
                                                                 boxb).area()

    @staticmethod
    def from_points(p0, p1):
        """
        Convert main corner points to bounding box.

        >> p0: The left-top points in (x, y).
        >> p1: The right-bottom points in (x, y).

        << The instance of `Bbox`.

        ^^ AssertionError: if width or height is less than 0.
        """
        assert p1[0] >= p0[0], "width {} must larger than 0.".format(p1[0] -
                                                                     p0[0])
        assert p1[1] >= p0[1], "height {} must larger than 0.".format(p1[1] -
                                                                      p0[1])

        return Bbox(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1])


def two_dimension_sort_box(box1: Bbox, box2: Bbox, vratio=0.5):
    """bbox 两纬度排序

    Args:
        box1 (Bbox): [bbox1]
        box2 (Bbox): [bbox2]
        vratio (float, optional): [description]. Defaults to 0.5.

    Returns:
        [type]: [description]
    """
    kernel = [box1.left - box2.left, box1.top - box2.top]
    if box1.voverlap(box2) < vratio * min(box1.height, box2.height):
        kernel = [box1.top - box2.top, box1.left - box2.left]
    return kernel[0] if kernel[0] != 0 else kernel[1]


def two_dimension_sort_layout(layout1, layout2, vratio=0.54):
    """layout 排序
    """
    return two_dimension_sort_box(layout1["bbox"], layout2["bbox"])


def ppocr2example(ocr_res, img_path, simplify=True, scale=False):
    """Transfer paddleocr result to example
    """
    # 1. 将ocr输出结果从左上至右下排序
    segments = []
    for rst in ocr_res:
        left = min(rst[0][0][0], rst[0][3][0])
        top = min(rst[0][0][-1], rst[0][1][-1])
        width = max(rst[0][1][0], rst[0][2][0]) - min(rst[0][0][0],
                                                      rst[0][3][0])
        height = max(rst[0][2][-1], rst[0][3][-1]) - min(
            rst[0][0][-1], rst[0][1][-1])
        segments.append({
            "bbox": Bbox(*[left, top, width, height]),
            "text": rst[-1][0]
        })
    segments.sort(key=cmp_to_key(two_dimension_sort_layout))
    # 2. im_base64
    img_base64 = img2base64(img_path)
    # 3. doc_tokens, doc_boxes, segment_ids
    doc_tokens = []
    doc_boxes = []

    im_w_box = max([seg["bbox"].left + seg["bbox"].width
                    for seg in segments]) + 20
    im_h_box = max([seg["bbox"].top + seg["bbox"].height
                    for seg in segments]) + 20
    img = Image.open(img_path)
    im_w, im_h = img.size  # 图片的实际大小
    im_w, im_h = max(im_w, im_w_box), max(im_h, im_h_box)

    if scale:
        image_size = 1024
        scale_x = image_size / im_w
        scale_y = image_size / im_h
    for segment in segments:
        bbox = segment["bbox"]  # x, y, w, h
        x1, y1, w, h = bbox.left, bbox.top, bbox.width, bbox.height
        if scale:
            w = int(min(w * scale_x, image_size - 1))
            h = int(min(h * scale_y, image_size - 1))
            y1 = int(max(0, min(y1 * scale_y, image_size - h - 1)))
            x1 = int(max(0, min(x1 * scale_x, image_size - w - 1)))
        bbox = Bbox(*[x1, y1, w, h])
        text = segment["text"]
        char_num = 0
        eng_word = ""
        # 获取doc_tokens、doc_segment_ids
        for char in text:
            # 如果是中文，且不存在英文
            if not check(char) and not eng_word:
                doc_tokens.append(char)
                char_num += 1
            # 如果是中文，且存在英文
            elif not check(char) and eng_word:
                doc_tokens.append(eng_word)
                eng_word = ""
                doc_tokens.append(char)
                char_num += 2
            # 如果是英文
            else:
                eng_word += char
        if eng_word:
            doc_tokens.append(eng_word)
            char_num += 1
        # 获取doc_boxes
        char_width = int(w / char_num)
        for char_idx in range(char_num):
            doc_boxes.append([
                Bbox(*[
                    bbox.left +
                    (char_width * char_idx), bbox.top, char_width, bbox.height
                ])
            ])
    new_doc_boxes = []
    if simplify:
        for doc_box in doc_boxes:
            bbox = doc_box[0]
            new_doc_boxes.append([bbox.left, bbox.top, bbox.width, bbox.height])
        doc_boxes = new_doc_boxes
    example = {
        "text": doc_tokens,
        "bbox": doc_boxes,
        "width": im_w,
        "height": im_h,
        "image": img_base64
    }
    return example
