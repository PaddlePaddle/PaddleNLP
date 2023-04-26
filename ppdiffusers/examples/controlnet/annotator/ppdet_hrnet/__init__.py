# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import cv2
import numpy as np
import paddle
import paddlehub as hub

from . import util
from .det_keypoint_unite_infer import PPDetPose


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
            result = self.ppdetpose_pred(oriImg)
            result["candidate"] = result["candidate"] * img_scalarfactor
            oriImg = cv2.resize(oriImg, (0, 0), fx=img_scalarfactor, fy=img_scalarfactor)
            canvas = oriImg.copy()
            canvas.fill(0)
            canvas = self.body_estimation.draw_pose(canvas, result["candidate"], result["subset"])
            if hand:
                hands_list = util.hand_detect(result["candidate"], result["subset"], oriImg)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    scale_search = [x * img_scalarfactor for x in [0.5, 1.0, 1.5, 2.0]]
                    peaks = self.hand_estimation.hand_estimation(
                        oriImg[y : y + w, x : x + w, ::-1], scale_search=scale_search
                    )
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)

            return canvas, dict(candidate=result["candidate"].tolist(), subset=result["subset"].tolist())

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
        return {"candidate": candidate, "subset": subset}
