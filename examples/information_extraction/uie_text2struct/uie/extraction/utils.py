#!/usr/bin/env python
# -*- coding:utf-8 -*-


def convert_spot_asoc(spot_asoc_instance, structure_maker):
    """Convert spot asoc instance to target string
    """
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['label'],
            structure_maker.target_span_start,
            spot['span'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_label,
                structure_maker.target_span_start,
                asoc_span,
                structure_maker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [
            ' '.join([
                structure_maker.record_start,
                ' '.join(spot_str_rep),
                structure_maker.record_end,
            ])
        ]
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])
    return target_text


convert_to_record_function = {'spotasoc': convert_spot_asoc, }
