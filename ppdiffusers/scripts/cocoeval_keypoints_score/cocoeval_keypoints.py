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

import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# This script references https://cocodataset.org/#keypoints-eval.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gt", type=str, help="Assign the groud true path.", default=None)
    parser.add_argument("-d", "--dt", type=str, help="Assign the detection result path.", default=None)
    args = parser.parse_args()

    cocoGt = COCO(args.gt)
    cocoDt = cocoGt.loadRes(args.dt)
    cocoEval = COCOeval(cocoGt, cocoDt, "keypoints")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
