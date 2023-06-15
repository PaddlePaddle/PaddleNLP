# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import json
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import cv2
import numpy as np
import paddle
import paddlehub as hub
from annotator.ppdet_hrnet.det_keypoint_unite_infer import PPDetPose

# import PIL
from PIL import Image
from tqdm import tqdm


def keypoint_to_openpose_kpts(coco_keypoints_list):
    # coco keypoints: [x1,y1,v1,...,xk,yk,vk]       (k=17)
    #     ['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
    #      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
    # openpose keypoints: [y1,...,yk], [x1,...xk]   (k=18, with Neck)
    #     ['Nose', *'Neck'*, 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip',
    #      'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Reye', 'Leye', 'Rear', 'Lear']
    indices = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    openpose_kpts = []
    for i in indices:
        openpose_kpts.append(coco_keypoints_list[i])

    # Get 'Neck' keypoint by interpolating between 'Lsho' and 'Rsho' keypoints
    l_shoulder_index = 5
    r_shoulder_index = 6
    l_shoulder_keypoint = coco_keypoints_list[l_shoulder_index]
    r_shoulder_keypoint = coco_keypoints_list[r_shoulder_index]

    neck_keypoint_y = int((l_shoulder_keypoint[1] + r_shoulder_keypoint[1]) / 2.0)
    neck_keypoint_x = int((l_shoulder_keypoint[0] + r_shoulder_keypoint[0]) / 2.0)
    neck_keypoint = [neck_keypoint_x, neck_keypoint_y, min(l_shoulder_keypoint[2], r_shoulder_keypoint[2])]
    open_pose_neck_index = 1
    openpose_kpts.insert(open_pose_neck_index, neck_keypoint)

    return openpose_kpts


class PPDetDetector:
    def __init__(self):
        self.body_estimation = hub.Module(name="openpose_body_estimation")
        self.hand_estimation = hub.Module(name="openpose_hands_estimation")
        self.ppdetpose = PPDetPose()

    def __call__(self, oriImg, detect_resolution=512, hand=False):
        with paddle.no_grad():
            img_scalarfactor = detect_resolution / min(oriImg.shape[:2])
            result, poseres = self.ppdetpose_pred(oriImg)
            result["candidate"] = result["candidate"] * img_scalarfactor
            oriImg = cv2.resize(oriImg, (0, 0), fx=img_scalarfactor, fy=img_scalarfactor)
            canvas = oriImg.copy()
            canvas.fill(0)
            canvas = self.body_estimation.draw_pose(canvas, result["candidate"], result["subset"])

            return canvas, dict(candidate=result["candidate"].tolist(), subset=result["subset"].tolist()), poseres

    def ppdetpose_pred(self, image, kpt_threshold=0.3):
        poseres = self.ppdetpose.ppdet_hrnet_infer(image)
        keypoints = poseres["keypoint"][0]
        num_kpts = len(keypoints)
        subset = np.ones((num_kpts, 20)) * -1
        candidate = np.zeros((0, 4))
        posnum = 0
        for kptid, keypoint in enumerate(keypoints):
            openpose_kpts = keypoint_to_openpose_kpts(keypoint)
            for idx, item in enumerate(openpose_kpts):
                if item[2] > kpt_threshold:
                    subset[kptid][idx] = posnum
                    kpt = np.array(
                        item
                        + [
                            posnum,
                        ]
                    )
                    candidate = np.vstack((candidate, kpt))
                    posnum += 1
        return {"candidate": candidate, "subset": subset}, poseres


annotator_ckpts_path = os.path.join(os.path.dirname(__file__), "ckpts")


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def get_keypoints_result_coco_format(paths, detector, do_gt):
    """Get keypoints result in coco format"""
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])
    in_dir_path = pathlib.Path(paths[0])
    if len(paths) == 3:
        out_dir_path = pathlib.Path(paths[2])
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in in_dir_path.glob("*.{}".format(ext))])
    output = []
    index = -1
    for file in tqdm(files):
        index += 1
        im = Image.open(file)
        im = np.array(im, dtype=np.uint8)
        input_image = HWC3(im)
        canvas, keypoints_result, poseres = detector(input_image)
        if len(paths) == 3:
            Image.fromarray(canvas).save(os.path.join(out_dir_path, os.path.basename(file)))
        if len(poseres["keypoint"][0]) == 0:
            sample_dict = {
                "image_id": index,
                "category_id": 1,
                "keypoints": [0, 0, 0] * 17,
                "score": 0,
                "id": index,
                "num_keypoints": 0,
                "bbox": [0, 0, 0, 0],
                "area": 0,
                "iscrowd": 0,
            }
        else:
            keypoints_list = []
            zero_num = 0
            for point in poseres["keypoint"][0][0]:
                if point[2] < 0.3:
                    keypoints_list += [0, 0, 0]
                    zero_num += 1
                else:
                    keypoints_list += point[:2] + [2]

            sample_dict = {
                "image_id": index,
                "category_id": 1,
                "keypoints": keypoints_list,
                "score": poseres["keypoint"][1][0][0],
                "id": index,
                "num_keypoints": 17 - zero_num,
                "bbox": poseres["bbox"][0],
                "area": poseres["bbox"][0][2] * poseres["bbox"][0][3],
                "iscrowd": 0,
            }

        output.append(sample_dict)

    with open(paths[1], "w") as json_file:
        if do_gt:
            json_file.write(
                json.dumps(
                    {
                        "annotations": output,
                        "images": [{"id": item} for item in list(range(index + 1))],
                        "categories": [
                            {
                                "supercategory": "person",
                                "id": 1,
                                "name": "person",
                                "keypoints": [
                                    "nose",
                                    "left_eye",
                                    "right_eye",
                                    "left_ear",
                                    "right_ear",
                                    "left_shoulder",
                                    "right_shoulder",
                                    "left_elbow",
                                    "right_elbow",
                                    "left_wrist",
                                    "right_wrist",
                                    "left_hip",
                                    "right_hip",
                                    "left_knee",
                                    "right_knee",
                                    "left_ankle",
                                    "right_ankle",
                                ],
                                "skeleton": [
                                    [16, 14],
                                    [14, 12],
                                    [17, 15],
                                    [15, 13],
                                    [12, 13],
                                    [6, 12],
                                    [7, 13],
                                    [6, 7],
                                    [6, 8],
                                    [7, 9],
                                    [8, 10],
                                    [9, 11],
                                    [2, 3],
                                    [1, 2],
                                    [1, 3],
                                    [2, 4],
                                    [3, 5],
                                    [4, 6],
                                    [5, 7],
                                ],
                            }
                        ],
                    },
                    indent=4,
                )
            )
        else:
            json_file.write(json.dumps(output, indent=4))


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--do_gt", action="store_true", help="whether to predict unseen future data")
parser.add_argument(
    "path", type=str, nargs=3, help=("Paths to the input images dir, output json file, and output openpose images dir")
)

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

if __name__ == "__main__":
    args = parser.parse_args()
    detector = PPDetDetector()
    get_keypoints_result_coco_format(args.path, detector, args.do_gt)
