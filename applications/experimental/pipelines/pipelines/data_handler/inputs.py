# Copyright 2021 deepset GmbH. All Rights Reserved.
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

from typing import List, Union


class Question:

    def __init__(self, text: str, uid: str = None):
        self.text = text
        self.uid = uid

    def to_dict(self):
        ret = {"question": self.text, "id": self.uid, "answers": []}
        return ret


class QAInput:

    def __init__(self, doc_text: str, questions: Union[List[Question],
                                                       Question]):
        self.doc_text = doc_text
        if type(questions) == Question:
            self.questions = [questions]
        else:
            self.questions = questions  # type: ignore

    def to_dict(self):
        questions = [q.to_dict() for q in self.questions]
        ret = {"qas": questions, "context": self.doc_text}
        return ret
