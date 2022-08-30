# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse

from pydub import AudioSegment

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert other audio formats to wav.')
    parser.add_argument('--audio_file',
                        type=str,
                        required=True,
                        help='Path to source audio.')
    parser.add_argument('--wav_file',
                        type=str,
                        default=None,
                        help='Path to save .wav file.')
    parser.add_argument('--audio_format',
                        type=str,
                        default=None,
                        help='The file extension.')
    args = parser.parse_args()

    supported = ['mp3', 'm4a', 'wav']
    if args.audio_format is not None:
        if args.audio_format not in supported:
            raise ValueError('.%s format file is not supported!' %
                             args.audio_format)
        supported = [args.audio_format]
    print('All %s format files are converted to .wav format...' %
          ', '.join(supported))

    if os.path.isfile(args.audio_file):
        src_files = [args.audio_file]
        if args.audio_format is not None:
            if args.audio_format != args.audio_file.strip().split('.')[-1]:
                raise ValueError(
                    'Ignore audio_format %s! It is not consistent with the format of audio_file %s.'
                    % (args.audio_format, args.audio_file))
    elif os.path.isdir(args.audio_file):
        src_files = [
            x for x in os.listdir(args.audio_file)
            if x.strip().split('.')[-1] in supported
        ]
        src_files = [os.path.join(args.audio_file, x) for x in src_files]
    else:
        raise Exception('%s is neither valid path nor file!' % args.audio_file)

    if args.wav_file is None:
        wav_files = [os.path.basename(x)[:-3] + 'wav' for x in src_files]
    elif os.path.isfile(args.wav_file):
        if len(src_files) == 1:
            wav_files = [args.wav_file]
        else:
            raise Exception('All audios in %s will overwrite the same file %s! \
                Please check it.' % (args.audio_file, args.wav_file))
    else:
        if not os.path.exists(args.wav_file):
            os.makedirs(args.wav_file)
        wav_files = [
            os.path.join(args.wav_file,
                         os.path.basename(x)[:-3] + 'wav') for x in src_files
        ]

    for src_file, wav_file in zip(src_files, wav_files):
        audio = AudioSegment.from_file(src_file, src_file[-3:])
        wav_audio = audio.export(wav_file, format='wav')

    print('%d files converted!' % len(src_files))
