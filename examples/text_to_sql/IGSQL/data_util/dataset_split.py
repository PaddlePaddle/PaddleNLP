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
"""Utility functions for loading and processing ATIS data."""

import os
import pickle


class DatasetSplit:
    """Stores a split of the ATIS dataset.

    Attributes:
        examples (`list`): Stores the examples in the split.
    """

    def __init__(self, processed_filename, raw_filename, load_function):
        if os.path.exists(processed_filename):
            print("Loading preprocessed data from " + processed_filename)
            with open(processed_filename, 'rb') as infile:
                self.examples = pickle.load(infile)
        else:
            print("Loading raw data from " + raw_filename + " and writing to " +
                  processed_filename)

            infile = open(raw_filename, 'rb')
            examples_from_file = pickle.load(infile)
            assert isinstance(examples_from_file, list), raw_filename + \
                " does not contain a list of examples"
            infile.close()

            self.examples = []
            for example in examples_from_file:
                obj, keep = load_function(example)

                if keep:
                    self.examples.append(obj)

            print("Loaded " + str(len(self.examples)) + " examples")
            outfile = open(processed_filename, 'wb')
            pickle.dump(self.examples, outfile)
            outfile.close()

    def get_ex_properties(self, function):
        """ Applies some function to the examples in the dataset.

        Args:
            function (`function`): Function to apply to all examples.

        Returns
            `list`: The return value of the function
        """
        elems = []
        for example in self.examples:
            elems.append(function(example))
        return elems
