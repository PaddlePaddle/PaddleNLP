import os
import json
from paddleocr import PaddleOCR
import numpy as np
import re
import time
import sys
from multiprocessing import Process, Queue

from paddlenlp.transformers import LayoutXLMTokenizer

tokenizer = LayoutXLMTokenizer.from_pretrained("layoutxlm-base-uncased")


def get_all_chars(tokenizer):
    all_chr = []
    for i in range(30000):
        tok_chr = tokenizer.tokenize(chr(i))
        tok_chr = [tc.replace("▁", "") for tc in tok_chr]
        while "" in tok_chr:
            tok_chr.remove("")
        tok_chr = ''.join(tok_chr)
        if len(tok_chr) != 1:
            all_chr.append(i)
    return all_chr


def merge_bbox(tok_bboxes):
    min_gx = min([box[0] for box in tok_bboxes])
    max_gx = max([box[1] for box in tok_bboxes])
    min_gy = min([box[2] for box in tok_bboxes])
    max_gy = max([box[3] for box in tok_bboxes])
    height_g = max_gy - min_gy
    width_g = max_gx - min_gx
    height_m = 0
    width_m = 0
    for box in tok_bboxes:
        x_min, x_max, y_min, y_max = box
        height_m += y_max - y_min
        width_m += x_max - x_min
    height_m = height_m / len(tok_bboxes)
    if (height_g -
            height_m) < 0.5 * height_m and width_g - width_m < 0.1 * width_m:
        return False, [min_gx, max_gx, min_gy, max_gy]
    else:
        return True, tok_bboxes[0]


def xlm_parse(ocr_res, tokenizer):

    doc_tokens, doc_bboxes = [], []
    all_chr = get_all_chars(tokenizer)

    try:
        new_tokens, new_token_boxes = [], []
        for item in ocr_res:
            new_tokens.extend(item["tokens"])
            new_token_boxes.extend(item["token_box"])

        # get layoutxlm tokenizer results and get the final results
        temp_span_text = ''.join(new_tokens)
        temp_span_bbox = new_token_boxes
        span_text = ""
        span_bbox = []
        # drop blank space
        for text, bbox in zip(temp_span_text, temp_span_bbox):
            if text == " ":
                continue
            else:
                span_text += text
                span_bbox += [bbox]

        # span_tokens starts with "_"
        span_tokens = tokenizer.tokenize(span_text)
        span_tokens[0] = span_tokens[0].replace("▁", "")
        while "" in span_tokens:
            span_tokens.remove("")

        doc_bboxes = []
        i = 0
        for tid, tok in enumerate(span_tokens):
            tok = tok.replace("▁", "")
            if tok == "":
                doc_bboxes.append(span_bbox[i])
                continue
            if tok == "<unk>":
                if tid + 1 == len(span_tokens):
                    tok_len = 1
                else:
                    if span_tokens[tid + 1] == "<unk>":
                        tok_len = 1
                    else:
                        for j in range(i, len(span_text)):
                            if span_text[j].lower() == span_tokens[tid + 1][0]:
                                break
                        tok_len = j - i
            elif ord(span_text[i]) in all_chr:
                if tid + 1 == len(span_tokens):
                    tok_len = 1
                elif "°" in tok and "C" in span_tokens[tid + 1]:
                    tok_len = len(tok) - 1
                    if tok_len == 0:
                        doc_bboxes.append(span_bbox[i])
                        continue
                elif span_text[i] == "ⅱ":
                    if tok == "ii":
                        if span_text[i + 1] != "i":
                            tok_len = len(tok) - 1
                        else:
                            tok_len = len(tok)
                    elif tok == "i":
                        tok_len = len(tok) - 1
                        if tok_len == 0:
                            doc_bboxes.append(span_bbox[i])
                            continue
                elif "m" in tok and "2" == span_tokens[tid + 1][0]:
                    tok_len = len(tok) - 1
                    if tok_len == 0:
                        doc_bboxes.append(span_bbox[i])
                        continue
                elif ord(span_text[i + 1]) in all_chr:
                    tok_len = 1
                else:
                    for j in range(i, len(span_text)):
                        if span_text[j].lower() == span_tokens[tid + 1][0]:
                            break
                        if span_text[j].lower() == "，" and span_tokens[
                                tid + 1][0] == ",":
                            break
                        if span_text[j].lower() == "；" and span_tokens[
                                tid + 1][0] == ";":
                            break
                        if span_text[j].lower() == "）" and span_tokens[
                                tid + 1][0] == ")":
                            break
                        if span_text[j].lower() == "（" and span_tokens[
                                tid + 1][0] == "(":
                            break
                        if span_text[j].lower() == "￥" and span_tokens[
                                tid + 1][0] == "¥":
                            break

                    tok_len = j - i

            else:
                if "�" == span_text[i]:
                    tok_len = len(tok) + 1
                elif tok == "......" and "…" in span_text[i:i + 6]:
                    tok_len = len(tok) - 2
                elif "ⅱ" in span_text[i + len(tok) - 1]:
                    if tok == "i":
                        tok_len = 1
                    else:
                        tok_len = len(tok) - 1
                elif "°" in tok and "C" in span_tokens[tid + 1]:
                    tok_len = len(tok) - 1
                else:
                    tok_len = len(tok)

            assert i + tok_len <= len(span_bbox)
            tok_bboxes = span_bbox[i:i + tok_len]
            _, merged_bbox = merge_bbox(tok_bboxes)

            doc_bboxes.append(merged_bbox)
            i = i + tok_len
    except:
        print('Error')
        span_tokens = ['▁'] * 512
        doc_bboxes = [[0, 0, 0, 0]] * 512

    return span_tokens, doc_bboxes


def tokenize_ocr_res(ocr_reses):
    '''
    input:
        ocr_res: the ocr result of the image
    return:
        new_reses: {
            pid: {
                "text": all text in each ocr_res,
                "bounding_box": the bounding box of the ocr_res,
                "tokens": all chars in ocr_res,
                "token_box: bounding box of each chars in ocr_res
                }
        }
    '''
    new_reses = []
    for img_name, ocr_res in ocr_reses:
        new_res = []
        for para in ocr_res:
            text = para["text"]
            text_box = para["bbox"]
            x_min, y_min = [int(min(idx)) for idx in zip(*text_box)]
            x_max, y_max = [int(max(idx)) for idx in zip(*text_box)]
            text_chars = list(text.lower())
            char_num = 0
            for char in text_chars:
                if re.match("[^\x00-\xff]", char):
                    char_num += 2
                else:
                    char_num += 1
            width = x_max - x_min
            shift = x_min
            new_token_boxes, new_tokens = [], []
            for char in text_chars:
                if re.match("[^\x00-\xff]", char):
                    tok_x_max = shift + width / char_num * 2
                else:
                    tok_x_max = shift + width / char_num * 1
                tok_x_min = shift
                tok_y_min = y_min
                tok_y_max = y_max

                shift = tok_x_max
                new_token_boxes.append(
                    [round(tok_x_min),
                     round(tok_x_max), tok_y_min, tok_y_max])
                new_tokens.append(char)
            new_res.append({
                "text": para["text"],
                "bounding_box": para["bbox"],
                "tokens": new_tokens,
                "token_box": new_token_boxes
            })
        new_reses.append((img_name, new_res))
    return new_reses


def process_input(ocr_reses, tokenizer, save_ocr_path):
    ocr_reses = tokenize_ocr_res(ocr_reses)

    examples = []
    for img_name, ocr_res in ocr_reses:
        doc_tokens, doc_bboxes = xlm_parse(ocr_res, tokenizer)
        doc_tokens.insert(0, '▁')
        doc_bboxes.insert(0, doc_bboxes[0])
        example = {
            "img_name": img_name,
            "document": doc_tokens,
            "document_bbox": doc_bboxes
        }
        examples.append(example)

    with open(save_ocr_path, 'w', encoding='utf8') as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')

    print(f"ocr parsing results has been save to: {save_ocr_path}")


def ocr_preprocess(img_dir):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
    ocr_reses = []
    img_names = sorted(os.listdir(img_dir),
                       key=lambda x: int(x.split("_")[1].split(".")[0]))
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        parsing_res = ocr.ocr(img_path, cls=True)
        ocr_res = []
        for para in parsing_res:
            ocr_res.append({"text": para[1][0], "bbox": para[0]})
        ocr_reses.append((img_name, ocr_res))

    return ocr_reses


if __name__ == '__main__':
    img_dir = "./demo_pics"
    save_path = "./demo_ocr_res.json"
    ocr_results = ocr_preprocess(img_dir)
    process_input(ocr_results, tokenizer, save_path)
