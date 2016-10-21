import numpy as np
import cv2
import time
import psegutils
import argparse
import sys
import os
import ctypes
import common


lib = ctypes.cdll.LoadLibrary(sys.path[0] + '/layout/libRLSA.so')


def horizontal_smear(binary, horizontal_threshold=20):
    lib.horizontal_smear(ctypes.c_void_p(binary.ctypes.data), ctypes.c_int(binary.shape[0]),
                         ctypes.c_int(binary.shape[1]), horizontal_threshold)


def vertical_smear(binary, vertical_threshold=30):
    lib.vertical_smear(ctypes.c_void_p(binary.ctypes.data), ctypes.c_int(binary.shape[0]),
                       ctypes.c_int(binary.shape[1]), vertical_threshold)


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


def draw_layouts(input_path, boxes, output_path):
    img = cv2.imread(input_path)
    for box in boxes:
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(output_path, img)


def process_args(args):
    parser = argparse.ArgumentParser("""
    Image layout using Run-Length Smearing Algorithm.
    """)
    parser.add_argument('files', nargs='+')
    parser.add_argument('-d', '--debug', action="store_true", help='display intermediate results')
    args = parser.parse_args(args)
    args.files = common.glob_all(args.files)
    if len(args.files) < 1:
        parser.print_help()
        sys.exit(0)
    return args


# ==== Page layout analyze ====
def layout(img_path, args=None):
    """
    :param img_path: binary image path
    :param args: for debug or other options
    :return: box list (x,y,w,h)
    """
    img_start_time = prev_time = time.time()
    base_path = os.path.splitext(img_path)[0]
    if args is not None:
        debug = args.debug
    else:
        debug = False

    # Step1: read binary image
    binary = common.read_image_binary(img_path)
    print 'read image binary: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()

    # Step2: Estimate character scale
    character_scale = psegutils.estimate_scale(binary)
    print 'character_scale: {:.3f} {:.3f}s'.format(character_scale, time.time() - prev_time)
    prev_time = time.time()

    # Step3: Run-Length Smearing Algorithm
    assert character_scale > 0
    horizontal_smear(binary, int(round(character_scale * 3)))
    print 'horizontal_smear: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()
    vertical_smear(binary, int(round(character_scale * 3)))
    print 'vertical_smear: {:.3f}s'.format(time.time() - prev_time)
    prev_time = time.time()
    if debug:
        common.write_image_binary(base_path + ".binary.png", binary)

    # Step4: Find connected area
    boxes = find_connected_block(binary)
    print 'find connected block: {:.3f}s'.format(time.time() - prev_time)
    if debug:
        draw_layouts(img_path, boxes, base_path + ".boxes.png")
    print 'took: {:.3f}s \n'.format(time.time() - img_start_time)
    return boxes


def main(args):
    args = process_args(args)
    start_time = time.time()
    for img_path in args.files:
        layout(img_path, args)
    print 'totally took: {:.3f}s'.format(time.time() - start_time)
    print 'finish.'


if __name__ == "__main__":
    main(sys.argv[1:])
