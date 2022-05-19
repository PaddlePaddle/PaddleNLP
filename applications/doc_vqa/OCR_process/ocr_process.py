import os
import re
import sys
import json
import time
import numpy as np
from multiprocessing import Process, Queue

from paddleocr import PaddleOCR
from paddlenlp.transformers import LayoutXLMTokenizer

sys.path.insert(0, "../Extraction")
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
    if (height_g - height_m
        ) < 0.5 * height_m and width_g - width_m < 0.1 * width_m:
        return False, [min_gx, max_gx, min_gy, max_gy]
    else:
        return True, tok_bboxes[0]


def xlm_parse(new_paragraphs, ocr_index, tokenizer, q):
    ocr_res = new_paragraphs[ocr_index]
    labels = []
    doc_tokens, doc_bboxes = [], []
    all_ans_tokens = []
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
        for text, bbox in zip(temp_span_text, temp_span_bbox):
            if text == " ":
                continue
            else:
                span_text += text
                span_bbox += [bbox]
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
                        if span_text[j].lower() == "，" and span_tokens[tid + 1][
                                0] == ",":
                            break
                        if span_text[j].lower() == "；" and span_tokens[tid + 1][
                                0] == ";":
                            break
                        if span_text[j].lower() == "）" and span_tokens[tid + 1][
                                0] == ")":
                            break
                        if span_text[j].lower() == "（" and span_tokens[tid + 1][
                                0] == "(":
                            break
                        if span_text[j].lower() == "￥" and span_tokens[tid + 1][
                                0] == "¥":
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
        all_ans_tokens = ['']

    labels = ['O'] * len(span_tokens)

    q.put((ocr_index, [span_tokens, doc_bboxes, all_ans_tokens, labels]),
          block=False)


def tokenize_ocr_res(ocr_reses):
    '''
    input:
        ocr_res: the ocr result of the image
    return:
        new_paragraphs: {
            pid: {
                "text": all text in each paragraph,
                "bounding_box": the bounding box of the paragraph,
                "tokens": all chars in paragraph,
                "token_box: bounding box of each chars in paragraph
                }
        }
    '''
    new_paragraphs = []
    for ocr_res in ocr_reses:
        new_paragraph = []
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
                    [round(tok_x_min), round(tok_x_max), tok_y_min, tok_y_max])
                new_tokens.append(char)
            new_paragraph.append({
                "text": para["text"],
                "bounding_box": para["bbox"],
                "tokens": new_tokens,
                "token_box": new_token_boxes
            })
        new_paragraphs.append(new_paragraph)
    return new_paragraphs


def process_multi_page(ocr_reses):
    for ocr_res in ocr_reses:
        min_page_id = 100000
        max_page_height = -1
        for para in ocr_res:
            page_id = int(para['page_id'])
            if page_id < min_page_id:
                min_page_id = page_id
            xmin, xmax, ymin, ymax = [
                para['bbox'][0][0], para['bbox'][1][0], para['bbox'][0][1],
                para['bbox'][2][1]
            ]

            if ymax > max_page_height:
                max_page_height = ymax

        for para in ocr_res:
            para['bbox'][0][1] += max_page_height * (
                int(para['page_id']) - min_page_id)
            para['bbox'][1][1] += max_page_height * (
                int(para['page_id']) - min_page_id)
            para['bbox'][2][1] += max_page_height * (
                int(para['page_id']) - min_page_id)
            para['bbox'][3][1] += max_page_height * (
                int(para['page_id']) - min_page_id)

    return min_page_id, max_page_height


def process_input(ocr_reses, tokenizer, save_path):
    max_num = min(10, len(ocr_reses))
    max_height_list, min_page_num_list = [], []
    for ocr_res in ocr_reses:
        min_page_id = 100000
        max_page_height = -1
        for para in ocr_res:
            page_id = int(para['page_id'])
            if page_id < min_page_id:
                min_page_id = page_id
            xmin, xmax, ymin, ymax = [
                para['bbox'][0][0], para['bbox'][1][0], para['bbox'][0][1],
                para['bbox'][2][1]
            ]

            if ymax > max_page_height:
                max_page_height = ymax

        min_page_num_list.append(min_page_id)
        # for the text in the bottom, +1 ignores the calc mistakes
        max_height_list.append(max_page_height)

        for para in ocr_res:
            para['bbox'][0][1] += max_page_height * (
                int(para['page_id']) - min_page_id)
            para['bbox'][1][1] += max_page_height * (
                int(para['page_id']) - min_page_id)
            para['bbox'][2][1] += max_page_height * (
                int(para['page_id']) - min_page_id)
            para['bbox'][3][1] += max_page_height * (
                int(para['page_id']) - min_page_id)
    begin_time = time.time()
    new_paragraphs = tokenize_ocr_res(ocr_reses[:max_num])
    begin_time = time.time()
    input_examples = []
    process_list = []
    begin_time = time.time()
    q = Queue()
    for i in range(max_num):
        p = Process(target=xlm_parse, args=(new_paragraphs, i, tokenizer, q))
        p.start()
        process_list.append(p)

    xlm_res = {}
    for p in process_list:
        ocr_index, content = q.get()
        xlm_res[ocr_index] = content
    for i in range(len(xlm_res)):
        doc_tokens, doc_bboxes, all_ans_tokens, labels = xlm_res[i]
        doc_tokens.insert(0, '▁')
        doc_bboxes.insert(0, doc_bboxes[0])
        example = {
            "id": "71f5c6e586e230a231bfee7cd3194efc",
            "question": "",
            "image_id": "71f5c6e586e230a231bfee7cd3194efc",
            "url": "",
            "document": doc_tokens,
            "document_bbox": doc_bboxes,
            "answer": [""],
            "labels": labels
        }

        input_examples.append(example)

    for p in process_list:
        p.join()

    with open(save_path, 'w', encoding='utf8') as f:
        for line in input_examples:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')

    return input_examples, min_page_num_list, max_height_list


def ocr_preprocess(img_dir):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
    lines = []
    num = len(os.listdir(img_dir))
    for i in range(1, num + 1):
        filename = os.path.join(img_dir, f"demo_{i}.png")
        ocr_result = ocr.ocr(filename, cls=True)
        ocr_line = []
        for line in ocr_result:
            ocr_line.append({
                "text": line[1][0],
                "bbox": line[0],
                "page_id": "1".zfill(5)
            })
        lines.append(ocr_line)
    return lines


if __name__ == '__main__':
    img_dir = "./demo_pics"
    save_path = "./demo_ocr_res.json"
    ocr_reses = ocr_preprocess(img_dir)
    process_input(ocr_reses, tokenizer, save_path)
