# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import time
import numpy as np

from paddle_serving_server.pipeline import PipelineClient

client = PipelineClient()
client.connect(['127.0.0.1:8080'])

list_data = [
    "国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据",
    "试论翻译过程中的文化差异与语言空缺翻译过程,文化差异,语言空缺,文化对比"
]
feed = {}
for i, item in enumerate(list_data):
    feed[str(i)] = item

print(feed)
start_time = time.time()
ret = client.predict(feed_dict=feed)
end_time = time.time()
print("time to cost :{} seconds".format(end_time - start_time))

result = np.array(eval(ret.value[0]))
print(ret.key)
print(result.shape)
print(result)
