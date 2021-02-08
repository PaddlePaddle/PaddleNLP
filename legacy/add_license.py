import os

filePath = os.getcwd()


def get_all_files(dir):
    fileDirList = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            fileDirList.append(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            get_all_files(dir_path)

    return fileDirList


fileDirList = get_all_files(filePath)
for code in fileDirList:
    split = os.path.splitext(code)
    if (split[1] == '.py' and not '__init__' in split[0] and
            not '_ce' in split[0]):

        with open(code, 'r') as fz:
            content = fz.read()
        if content.find('Copyright') >= 0:
            fz.close()
            continue
        else:
            string = "#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.\n" \
                     "#\n" \
                     "# Licensed under the Apache License, Version 2.0 (the \"License\");\n" \
                     "# you may not use this file except in compliance with the License.\n" \
                     "# You may obtain a copy of the License at\n" \
                     "#\n" \
                     "#     http://www.apache.org/licenses/LICENSE-2.0\n" \
                     "#\n" \
                     "# Unless required by applicable law or agreed to in writing, software\n" \
                     "# distributed under the License is distributed on an \"AS IS\" BASIS,\n" \
                     "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n" \
                     "# See the License for the specific language governing permissions and\n" \
                     "# limitations under the License.\n"+content
            fz.close()
            with open(code, 'w') as f:
                f.write(string)
                print "file %s write success!" % code
            f.close()
print "read and write success!"
