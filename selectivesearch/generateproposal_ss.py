# %% 
from cmath import rect
import selectivesearch
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle
from uuid import uuid4
import numpy as np
import xml.etree.ElementTree as ET


def draw_region(img, regions):
    # candidates = set()
    # for r in regions:
    # excluding same rectangle (with different segments)
    # if r in candidates:
    #     continue
    # # excluding regions smaller than 2000 pixels
    # if r['size'] < 500:
    #     continue
    # distorted rects
    # if w / h > 1.2 or h / w > 1.2:
    #     continue
    # candidates.add(r)

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    regions = set(list(regions)[:3000])
    for x, y, w, h in regions:
        # print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


def save_object(obj, file_name, pickle_format=2):
    """Save a Python object by pickling it.
    Unless specifically overridden, we want to save it in Pickle format=2 since this
    will allow other Python2 executables to load the resulting Pickle. When we want
    to completely remove Python2 backward-compatibility, we can bump it up to 3. We
    should never use pickle.HIGHEST_PROTOCOL as far as possible if the resulting
    file is manifested or used, external to the system.
    """
    file_name = os.path.abspath(file_name)
    # Avoid filesystem race conditions (particularly on network filesystems)
    # by saving to a random tmp file on the same filesystem, and then
    # atomically rename to the target filename.
    tmp_file_name = file_name + ".tmp." + uuid4().hex
    with open(tmp_file_name, 'wb') as f:
        pickle.dump(obj, f, pickle_format)
        f.flush()  # make sure it's written to disk
        os.fsync(f.fileno())
    os.rename(tmp_file_name, file_name)


def get_minsize(xml_path):
    doc = ET.parse(xml_path)
    root = doc.getroot().findall('object')
    temp_size = []
    bbox = []
    for r in root:
        bb = r.find('bndbox')
        box = [
            bb.find("xmin").text,
            bb.find("ymin").text,
            bb.find("xmax").text,
            bb.find("ymax").text,
        ]
        box = list(map(int, box))
        bbox.append(box)
        temp_size.append((box[3] - box[1]) * (box[2] - box[0]))
    print(temp_size)
    # min_gt_size = int(np.mean(temp_size))
    min_gt_size = min(temp_size)
    return min_gt_size, bbox


boxes = []
ids = []
scale = 25
sigma = 0.8
min_size = 20

# label = 'trainval'
label = 'test'

with open('../data/VOCdevkit/VOC2007/ImageSets/Main/' + label + '.txt', 'r') as f:
    filelist = f.readlines()
# print(filelist)

for file in filelist:
    try:
        file = file.strip('\n')
        print('file', file)

        img = imread('../data/VOCdevkit/VOC2007/JPEGImages/' + file + '.jpg')

        min_gt_size, bbox = get_minsize('../data/VOCdevkit/VOC2007/Annotations/' + file + '.xml')
        if min(img.shape[0], img.shape[1]) >= 3000:
            min_size = max(int(min_gt_size // 1.5), 10)
            scale = 50
        elif min(img.shape[0], img.shape[1]) >= 2000:
            min_size = max(int(min_gt_size // 2), 10)
            scale = 25
        elif min(img.shape[0], img.shape[1]) >= 1000:
            min_size = max(min_gt_size // 5, 10)
            scale = 10
        elif min(img.shape[0], img.shape[1]) >= 500:
            min_size = max(min_gt_size // 10, 10)
            scale = 5
        else:
            min_size = max(min_gt_size // 20, 10)
            scale = 1
        print('scale', scale, 'sigma', sigma, 'min_size', min_size)
        img_lbl, regions = selectivesearch.selective_search(img, scale, sigma, min_size)

        box = list(map(list, regions))
        if len(box) >= 3000:
            box = box[:3000]
        # box.extend(bbox) # 添加groundtruth
        print('num', len(box))
        # print(box)
        boxes.append(np.array(box).astype(np.float32))
        ids.append(int(file.split('.')[0]))
    except Exception as e:
        print(file, e)

file_name = 'SS_' + label + '_boxes.pkl'
print(file_name)
save_object(dict(boxes=boxes, indexes=ids), file_name)
