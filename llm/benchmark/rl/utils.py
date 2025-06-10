# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass
from typing import List


@dataclass
class RangeSet:
    """Manage processed line ranges with efficient storage and querying"""

    ranges: List[tuple]

    def add(self, number: int):
        """Add a number to the range set and merge adjacent ranges"""
        new_ranges = []
        added = False
        for start, end in sorted(self.ranges):
            if number < start - 1:
                if not added:
                    new_ranges.append((number, number))
                    added = True
                new_ranges.append((start, end))
            elif number == start - 1:
                new_ranges.append((number, end))
                added = True
            elif number <= end:
                new_ranges.append((start, end))
                added = True
            else:
                new_ranges.append((start, end))
        if not added:
            new_ranges.append((number, number))
        self.ranges = self.merge_ranges(new_ranges)

    @staticmethod
    def merge_ranges(ranges: List[tuple]) -> List[tuple]:
        """Merge overlapping or adjacent ranges"""
        if not ranges:
            return []
        sorted_ranges = sorted(ranges)
        merged = [sorted_ranges[0]]
        for current in sorted_ranges[1:]:
            last = merged[-1]
            if current[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        return merged

    def contains(self, number: int) -> bool:
        """Check if a number exists in any range"""
        for start, end in self.ranges:
            if start <= number <= end:
                return True
        return False

    def to_file_format(self) -> str:
        """Serialize ranges to compact string format"""
        return ",".join(f"{start}-{end}" if start != end else str(start) for start, end in self.ranges)

    @classmethod
    def from_file(cls, content: str) -> "RangeSet":
        """Deserialize from string format"""
        if not content:
            return cls(ranges=[])
        ranges = []
        for part in content.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                ranges.append((start, end))
            else:
                num = int(part)
                ranges.append((num, num))
        return cls(ranges=ranges)

    @property
    def processed_count(self) -> int:
        """Total number of processed items"""
        return sum(end - start + 1 for start, end in self.ranges)
