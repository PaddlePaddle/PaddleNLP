#!/usr/bin/env python
# -*- coding:utf-8 -*-
from uie.extraction.predict_parser.predict_parser import PredictParser
from uie.extraction.predict_parser.spotasoc_predict_parser import SpotAsocPredictParser

decoding_format_dict = {'spotasoc': SpotAsocPredictParser, }


def get_predict_parser(decoding_schema, label_constraint):
    return decoding_format_dict[decoding_schema](
        label_constraint=label_constraint)
