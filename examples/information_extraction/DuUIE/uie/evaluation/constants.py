#!/usr/bin/env python3
# -*- coding:utf-8 -*-

spot_prompt = '<spot>'
asoc_prompt = '<asoc>'

type_start = '<extra_id_0>'
type_end = '<extra_id_1>'
text_start = '<extra_id_2>'
span_start = '<extra_id_5>'
null_span = '<extra_id_6>'
null_lÍabel = '<extra_id_7>'

offset_map_strategy = {
    "closest_en": {
        "map_strategy": "closest",
        "de_duplicate": True,
        "span_to_token": "space",
    },
    "closest_zh": {
        "map_strategy": "closest",
        "de_duplicate": True,
        "span_to_token": "list",
    },
    "fisrt_en": {
        "map_strategy": "first",
        "de_duplicate": True,
        "span_to_token": "space",
    },
    "first_zh": {
        "map_strategy": "first",
        "de_duplicate": True,
        "span_to_token": "list",
    },
    "longer_first_zh": {
        "map_strategy": "longer_first",
        "de_duplicate": True,
        "span_to_token": "list",
    }
}


class BaseStructureMarker:
    def __init__(self) -> None:
        super().__init__()
        self.sent_start = '<extra_id_0>'
        self.sent_end = '<extra_id_1>'
        self.record_start = '<extra_id_0>'
        self.record_end = '<extra_id_1>'
        self.span_start = '<extra_id_0>'
        self.span_end = '<extra_id_1>'
        self.text_start = '<extra_id_2>'
        self.source_span_start = '<extra_id_3>'
        self.source_span_end = '<extra_id_4>'
        self.target_span_start = '<extra_id_5>'
        self.null_span = '<extra_id_6>'
        self.null_label = '<extra_id_7>'
