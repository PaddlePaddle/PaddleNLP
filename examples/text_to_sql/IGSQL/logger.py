#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Contains the logging class."""


class Logger():

    def __init__(self, filename, option):
        self.fileptr = open(filename, option)
        if option == "r":
            self.lines = self.fileptr.readlines()
        else:
            self.lines = []

    def put(self, string):
        """Writes to the file."""
        self.fileptr.write(string + "\n")
        self.fileptr.flush()

    def close(self):
        """Closes the logger."""
        self.fileptr.close()

    def findlast(self, identifier, default=0.):
        """Finds the last line in the log with a certain value."""
        for line in self.lines[::-1]:
            if line.lower().startswith(identifier):
                string = line.strip().split("\t")[1]
                if string.replace(".", "").isdigit():
                    return float(string)
                elif string.lower() == "true":
                    return True
                elif string.lower() == "false":
                    return False
                else:
                    return string
        return default

    def contains(self, string):
        """Dtermines whether the string is present in the log."""
        for line in self.lines[::-1]:
            if string.lower() in line.lower():
                return True
        return False

    def findlast_log_before(self, before_str):
        """Finds the last entry in the log before another entry."""
        loglines = []
        in_line = False
        for line in self.lines[::-1]:
            if line.startswith(before_str):
                in_line = True
            elif in_line:
                loglines.append(line)
            if line.strip() == "" and in_line:
                return "".join(loglines[::-1])
        return "".join(loglines[::-1])
