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
from annotator.mlsd import MLSDdetector
from annotator.util import HWC3, resize_image

apply_mlsd = MLSDdetector()


def process(input_image, image_resolution, detect_resolution):
    input_image = HWC3(input_image)
    detected_map = apply_mlsd(resize_image(input_image, detect_resolution), 0.1, 0.1)
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return 255 - cv2.dilate(detected_map, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)


input_image = cv2.imread("/paddle/PaddleDetection/demo/000000014439.jpg")
image_resolution = 512
detect_resolution = 512
output = process(input_image, image_resolution, detect_resolution)
cv2.imwrite("hough_output.jpg", output)
print("hough_output.jpg is saved")
