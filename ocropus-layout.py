import numpy as np
import ocrolib
import cv2


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


def horizontal_smear(binary):
    hor_thres = 40
    zero_count = 0
    one_flag = 0
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            val = binary[i][j]
            if val == 0:            # black
                if one_flag == 1:
                    if zero_count <= hor_thres:
                        binary[i][j] = 0
                        for n in range(j - zero_count, j):
                            binary[i][n] = 0
                    else:
                        one_flag = 0
                    zero_count = 0
                one_flag = 1
            else:                   # white
                zero_count += 1
    print 'horizontal_smear finish'


def vertical_smear(binary):
    ver_thres = 45
    zero_count = 0
    one_flag = 0
    for i in range(binary.shape[1]):
        for j in range(binary.shape[0]):
            val = binary[j][i]
            if val == 0:
                if one_flag == 1:
                    if zero_count <= ver_thres:
                        binary[j][i] = 0
                        for n in range(j - zero_count, j):
                            binary[n][i] = 0
                    else:
                        one_flag = 0
                    zero_count = 0
                one_flag = 1
            else:
                zero_count += 1
    print 'vertical_smear finish'


def find_connected_block(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for i, h in enumerate(hierarchy[0]):
        if h[3] == 0:      # parent is page
            # cv2.drawContours(img, contours, i, (0, 0, 255), 3)
            x, y, w, h = cv2.boundingRect(contours[i])
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            result.append((x, y, w, h))
    print 'find_connected_block: {}'.format(len(result))
    return result


def merge_overlap_boxes(boxes):
    union_boxes = []
    found_overlap = False
    while True:
        for i, box1 in enumerate(boxes):
            union_box = box1
            for box2 in boxes[i + 1:]:
                if is_overlap(box1, box2):
                    union_box = box_union(box1, box2)
                    found_overlap = True
                    break
            if union_box not in union_boxes:
                union_boxes.append(union_box)
        if found_overlap:   # prepare next loop
            boxes = union_boxes
            union_boxes = []
            found_overlap = False
        else:   # no overlap anymore
            break
    print 'merge_overlap_boxes: union_boxes({})'.format(len(union_boxes))
    return union_boxes


def draw_layouts(input_path, boxes, output_path):
    print 'draw_layouts...'
    img = cv2.imread(input_path)
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(output_path, img)


img_path = './camera/0001.bin.png'
out_path = './camera/0001.layout.png'
out_path2 = './camera/0001.layout2.png'
binary = ocrolib.read_image_binary(img_path)
horizontal_smear(binary)
vertical_smear(binary)
ocrolib.write_image_binary(out_path, binary)

boxes = find_connected_block(out_path)
union_boxes = merge_overlap_boxes(boxes)
draw_layouts(img_path, union_boxes, out_path2)


print 'finish.'
