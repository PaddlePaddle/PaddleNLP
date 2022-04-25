#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import json


def main():
    for line in sys.stdin:
        instance = json.loads(line.strip())
        new_instance = {
            'text': instance['text'],
            'entity': list(),
            'relation': list(),
            'event': list()
        }
        for ent in instance['extraction']['entity']['offset']:
            ent_type, offset = ent
            new_instance['entity'] += [{
                'offset': offset,
                'text': instance['text'][offset[0]:offset[-1] + 1],
                'type': ent_type
            }]
        for rel in instance['extraction']['relation']['offset']:
            rel_type, arg1_type, arg1_offset, arg2_type, arg2_offset = rel
            if len(arg1_offset) == 0 or len(arg2_offset) == 0:
                continue
            new_instance['relation'] += [{
                'type': rel_type,
                'args': [
                    {
                        'offset': arg1_offset,
                        'text':
                        instance['text'][arg1_offset[0]:arg1_offset[-1] + 1],
                        'type': arg1_type
                    },
                    {
                        'offset': arg2_offset,
                        'text':
                        instance['text'][arg2_offset[0]:arg2_offset[-1] + 1],
                        'type': arg2_type
                    },
                ]
            }]
        for evt in instance['extraction']['event']['offset']:
            valid_args = filter(lambda x: len(x[1]) > 0, evt['roles'])
            new_instance['event'] += [{
                'type': evt['type'],
                'offset': evt['trigger'],
                'text':
                instance['text'][evt['trigger'][0]:evt['trigger'][-1] + 1],
                'args': [{
                    'offset': arg[1],
                    'text': instance['text'][arg[1][0]:arg[1][-1] + 1],
                    'type': arg[0]
                } for arg in valid_args]
            }]
        # if len(new_instance['relation']) == 0 or len(new_instance['event']) == 0:
        #     continue
        print(json.dumps(new_instance, ensure_ascii=False))


if __name__ == "__main__":
    main()
