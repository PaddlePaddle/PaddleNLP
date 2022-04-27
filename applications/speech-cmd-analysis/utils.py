# coding=utf-8

import json
import time

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode


class ASRError(Exception):
    pass


def mandarin_asr_api(audio_file, audio_format='wav'):
    """ Mandarin ASR

    Args:
        audio_file (str):
            Audio file of Mandarin with sampling rate 16000.
        audio_format (str):
            The file extension of audio_file, 'wav' by default. 
    """
    # Configurations.
    TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
    ASR_URL = 'http://vop.baidu.com/server_api'
    SCOPE = 'audio_voice_assistant_get'
    API_KEY = 'vMkN1f7KlL3jornZnQGufBt4'
    SECRET_KEY = 'uHokaF8MM5YWi57dfvqCTpR0IaCXpaGB'
    
    # Fetch tokens from TOKEN_URL.
    post_data = urlencode({
        'grant_type': 'client_credentials',
        'client_id': API_KEY,
        'client_secret': SECRET_KEY}).encode('utf-8')

    request = Request(TOKEN_URL, post_data)
    try:
        result_str = urlopen(request).read()
    except URLError as error:
        print('token http response http code : ' + str(error.code))
        result_str = err.read()
    result_str = result_str.decode()

    result = json.loads(result_str)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if SCOPE and (not SCOPE in result['scope'].split(' ')):
            raise ASRError('scope is not correct!')
        token = result['access_token']
    else:
        raise ASRError('MAYBE API_KEY or SECRET_KEY not correct: ' + 
            'access_token or scope not found in token response')

    # Fetch results by ASR api.
    with open(audio_file, 'rb') as speech_file:
        speech_data = speech_file.read()
    length = len(speech_data)
    if length == 0:
        raise ASRError('file %s length read 0 bytes' % audio_file)
    params_query = urlencode({'cuid': 'ASR', 'token': token, 'dev_pid': 1537})
    headers = {
        'Content-Type': 'audio/%s; rate=16000' % audio_format,
        'Content-Length': length
    }

    url = ASR_URL + '?' + params_query
    request = Request(url, speech_data, headers)
    try:
        begin = time.time()
        result_str = urlopen(request).read()
        print('Request time cost %f' % (time.time() - begin))
    except  URLError as error:
        print('asr http response http code : ' + str(error.code))
        result_str = error.read()
    result_str = str(result_str, 'utf-8')
    result = json.loads(result_str)

    return result['result'][0]


