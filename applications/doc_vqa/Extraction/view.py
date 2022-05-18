import cv2
import json
import numpy as np


def view_ocr_result(img_path, bboxes, opath):
    image = cv2.imread(img_path)
    for char_bbox in bboxes:
        x_min, x_max, y_min, y_max = char_bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    cv2.imwrite(opath, image)


def _highlight_bbox(img, bbox):
    x = bbox[0]
    w = bbox[1] - x
    y = bbox[2]
    h = bbox[3] - y
    sub_img = img[y:y + h, x:x + w]
    colored_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    colored_rect[:, :, 2] = 255
    colored_rect[:, :, 1] = 255
    res = cv2.addWeighted(sub_img, 0.5, colored_rect, 0.5, 1.0)
    img[y:y + h, x:x + w] = res


def highlight_ans(source_img_path, output_img_path, ans_bbox):
    image = cv2.imread(source_img_path)
    for bbox in ans_bbox:
        _highlight_bbox(image, bbox)
    cv2.imwrite(output_img_path, image)


def highlight_img(source_img_path, output_img_path):
    image = cv2.imread(source_img_path)
    height = image.shape[0]
    width = image.shape[1]
    bbox = [0, width - 1, 0, height - 1]
    _highlight_bbox(image, bbox)
    cv2.imwrite(output_img_path, image)


if __name__ == '__main__':
    res_path = "./data/decode_res.json"
    result = {}
    with open(res_path, "r", encoding="utf-8") as f:
        line = f.readline()
        result = json.loads(line.strip())

    img_path = '../OCR_process/demo_pics/demo_{}.png'.format(result["img_id"])
    img_save_path = "../answer.png"
    highlight_ans(img_path, img_save_path, result['predict_bboxes'])
    print("extraction result has been saved to answer.png")
