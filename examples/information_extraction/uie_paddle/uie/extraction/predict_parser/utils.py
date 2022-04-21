#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re


def match_sublist(the_list, to_match):
    """

    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match:
        [1, 2]
        [1, 2, 4, 5]
    :return:
        [(0, 1), (6, 7)]
        [(6, 9)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def fix_unk_from_text(span, text, unk='<unk>', tokenizer=None):
    if tokenizer is not None:
        return fix_unk_from_text_with_tokenizer(
            span, text, unk=unk, tokenizer=tokenizer)
    else:
        return fix_unk_from_text_without_tokenizer(span, text, unk=unk)


def fix_unk_from_text_without_tokenizer(span, text, unk='<unk>'):
    """
    Find span from the text to fix unk in the generated span
    从 text 中找到 span，修复span

    Example:
    span = "<unk> colo e Bengo"
    text = "At 159 meters above sea level , Angola International Airport is located at Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "<unk> colo e Bengo"
    text = "Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "Arr<unk> s negre"
    text = "The main ingredients of Arròs negre , which is from Spain , are white rice , cuttlefish or squid , cephalopod ink , cubanelle and cubanelle peppers . Arròs negre is from the Catalonia region ."

    span = "colo <unk>"
    text = "At 159 meters above sea level , Angola International Airport is located at e Bengo , part of Luanda Province , in Angola . coloÍ"

    span = "Tarō As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "Tar<unk> As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "<unk>Tar As<unk>"
    text = "The leader of Japan is ōTar Asō ."
    """
    if unk not in span:
        return span

    def clean_wildcard(x):
        sp = ".*?()[]+"
        return re.sub("(" + "|".join([f"\\{s}" for s in sp]) + ")", "\\\\\g<1>",
                      x)

    match = r'\s*[^，？。\s]+\s*'.join(
        [clean_wildcard(item.strip()) for item in span.split(unk)])
    # match = r'\s*\S+\s*'.join([clean_wildcard(item.strip()) for item in span.split(unk)])
    result = re.search(match, text)

    if not result:
        return span
    return result.group().strip()


def fix_unk_from_text_with_tokenizer(span, text, tokenizer, unk='<unk>'):
    unk_id = tokenizer.vocab.to_indices(unk)
    # Paddle Tokenizer
    tokenized_span = tokenizer.encode(span)['input_ids'][:-1]
    tokenized_text = tokenizer.encode(text)['input_ids'][:-1]
    matched = match_sublist(tokenized_text, tokenized_span)

    if len(matched) == 0:
        raise RuntimeError("Cannot Match Sublist")

    if tokenized_span[0] == unk_id and matched[0][0] > 0:
        previous_token = [tokenized_text[matched[0][0] - 1]]
        pre_strip = tokenizer.vocab.to_tokens(previous_token[0])
    else:
        previous_token = []
        pre_strip = ""

    if tokenized_span[-1] == unk_id and matched[0][1] < len(tokenized_text) - 1:
        next_token = [tokenized_text[matched[0][1] + 1]]
        next_strip = tokenizer.vocab.to_tokens(next_token[0])
    else:
        next_token = []
        next_strip = ""

    extend_span = tokenized_span
    if len(previous_token) > 0:
        extend_span = previous_token + extend_span
    if len(next_token) > 0:
        extend_span = extend_span + next_token

    extend_span = tokenizer.decode(extend_span)
    fixed_span = fix_unk_from_text_without_tokenizer(extend_span, text, unk)
    return fixed_span.rstrip(next_strip).lstrip(pre_strip)


def test_fix_unk_from_text():

    span_text_list = [(
        "<unk> colo e Bengo",
        "At 159 meters above sea level , Angola International Airport is located at Ícolo e Bengo , part of Luanda Province , in Angola .",
        "Ícolo e Bengo"
    ), (
        "<unk> colo e Bengo",
        "Ícolo e Bengo , part of Luanda Province , in Angola .", "Ícolo e Bengo"
    ), (
        "Arr<unk> s negre",
        "The main ingredients of Arròs negre , which is from Spain , are white rice , cuttlefish or squid , cephalopod ink , cubanelle and cubanelle peppers . Arròs negre is from the Catalonia region .",
        "Arròs negre"
    ), (
        "colo <unk>",
        "At 159 meters above sea level , Angola International Airport is located at e Bengo , part of Luanda Province , in Angola . coloÍ",
        "coloÍ"
    ), ("Tarō As<unk>", "The leader of Japan is Tarō Asō .", "Tarō Asō"), (
        "Tar<unk> As<unk>", "The leader of Japan is Tarō Asō .", "Tarō Asō"
    ), ("<unk>Tar As<unk>", "The leader of Japan is ōTar Asō .", "ōTar Asō"), (
        "Atatürk Monument ( <unk> zmir )",
        "The Atatürk Monument ( İzmir ) can be found in Turkey .",
        "Atatürk Monument ( İzmir )"
    ), (
        "The Atatürk Monument [ <unk> zmir ]",
        "The Atatürk Monument [ İzmir ] can be found in Turkey .",
        "The Atatürk Monument [ İzmir ]"
    ), (
        "<unk>华泽二股东",
        "*1ST华泽二股东代表炮轰大股东",
        "*1ST华泽二股东",
    ), ("<unk>",
        "发表了博文《2014全球竞争力报告:瑞士再居首位+中 国升至28位》9月3日，总部位于日内瓦的世界经济论坛发布了《2014－2015年全球竞争力报告》，瑞士连续六年位居榜首，成为全球最具竞争力的国家，新http://t.cn/RhUNBxA",
        "发表了博文《2014全球竞争力报告:瑞士再居首位+中")]

    for span, text, gold in span_text_list:
        print(span, '|', fix_unk_from_text(span, text))
        assert fix_unk_from_text(span, text) == gold


if __name__ == "__main__":
    test_fix_unk_from_text()
