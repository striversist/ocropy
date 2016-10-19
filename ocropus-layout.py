import numpy as np
import ocrolib
import cv2
import time
import scipy.misc
from ocrolib import psegutils
import argparse
import sys
import os


parser = argparse.ArgumentParser("""
Image layout using Run-Length Smearing Algorithm.
""")
parser.add_argument('files', nargs='+')
parser.add_argument('-d', '--debug', action="store_true", help='display intermediate results')
args = parser.parse_args()
args.files = ocrolib.glob_all(args.files)
if len(args.files) < 1:
    parser.print_help()
    sys.exit(0)


def overlap(x1, w1, x2, w2):
    l1 = x1
    l2 = x2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1
    r2 = x2 + w2
    right = r1 if r1 < r2 else r2
    return right - left


def is_overlap(box1, box2):
    w = overlap(box1[0], box1[2], box2[0], box2[2])
    h = overlap(box1[1], box1[3], box2[1], box2[3])
    return False if (w < 0 or h < 0) else True


def box_union(box1, box2):
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[2], box2[2])
    h = max(box1[3], box2[3])
    return x, y, w, h


def horizontal_smear(binary, horizontal_threshold=20):
    zero_count = 0
    one_flag = 0
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            val = binary[i][j]
            if val == 0:            # black
                if one_flag == 1:
                    if zero_count <= horizontal_threshold:
                        binary[i][j] = 0
                        for n in range(j - zero_count, j):
                            binary[i][n] = 0
                    else:
                        one_flag = 0
                    zero_count = 0
                one_flag = 1
            else:                   # white
                zero_count += 1


def vertical_smear(binary, vertical_threshold=30):
    zero_count = 0
    one_flag = 0
    for i in range(binary.shape[1]):
        for j in range(binary.shape[0]):
            val = binary[j][i]
            if val == 0:
                if one_flag == 1:
                    if zero_count <= vertical_threshold:
                        binary[j][i] = 0
                        for n in range(j - zero_count, j):
                            binary[n][i] = 0
                    else:
                        one_flag = 0
                    zero_count = 0
                one_flag = 1
            else:
                zero_count += 1


def find_connected_block(binary):
    img = np.array(binary * 255, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for i, h in enumerate(hierarchy[0]):
        if h[3] != -1:      # no child
            # cv2.drawContours(img, contours, i, (0, 0, 255), 3)
            x, y, w, h = cv2.boundingRect(contours[i])
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            if w > 10 and h > 10:   # ignore too small
                result.append((x, y, w, h))
    print 'find_connected_block: {}'.format(len(result))
    return result


def resize_boxes(boxes, ratio_v, ratio_h):
    result = []
    for box in boxes:
        x = int(box[0] * ratio_h)
        y = int(box[1] * ratio_v)
        w = int(box[2] * ratio_h)
        h = int(box[3] * ratio_v)
        result.append((x, y, w, h))
    return result


def draw_layouts(input_path, boxes, output_path):
    img = cv2.imread(input_path)
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(output_path, img)


start_time = time.time()
for img_path in args.files:
    img_start_time = prev_time = time.time()
    base_path = os.path.splitext(img_path)[0]
    inter_shape = (1440, 1080)

    # Step1: read image and convert to binary
    gray_image = ocrolib.read_image_gray(img_path)
    print 'read image: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()

    # Step2: Resize to temp size for speeding up next processes
    ratio_vertical = gray_image.shape[0] * 1.0 / inter_shape[0]
    ratio_horizontal = gray_image.shape[1] * 1.0 / inter_shape[1]
    gray_image = scipy.misc.imresize(gray_image, inter_shape)
    binary = ocrolib.binarize_range(gray_image, dtype='i')
    print 'resize and binarize: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()

    # Step3: Estimate character scale
    character_scale = psegutils.estimate_scale(binary)
    print 'character_scale: {:.3f} {:.3f}s'.format(character_scale, time.time() - prev_time)
    prev_time = time.time()

    # Step4: Run-Length Smearing Algorithm
    assert character_scale > 0
    horizontal_smear(binary, int(round(character_scale * 3)))
    print 'horizontal_smear: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()
    vertical_smear(binary, int(round(character_scale * 3)))
    print 'vertical_smear: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()
    if args.debug:
        ocrolib.write_image_binary(base_path + ".binary.png", binary)

    # Step5: Find connected area
    boxes = find_connected_block(binary)
    print 'find connected block: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()

    # Step6: Resize boxes to original ratio
    boxes = resize_boxes(boxes, ratio_v=ratio_vertical, ratio_h=ratio_horizontal)
    print 'resize boxes: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()
    if args.debug:
        draw_layouts(img_path, boxes, base_path + ".boxes.png")
    print 'took: {:.3f}s \n'.format(time.time() - img_start_time)
print 'totally took: {:.3f}s'.format(time.time() - start_time)
print 'finish.'
