# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import os
import numpy as np
import skimage.data
from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


def main(img):
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=200, sigma=0.8, min_size=400)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 400 pixels
        if r['size'] < 400:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    list_all = []
    for x, y, w, h in candidates:
        list_one = [x, y, w, h]
        list_all.append(list_one)
        # print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()
    return list_all


if __name__ == "__main__":
    # loading astronaut image
    # image = skimage.data.astronaut()
    image = skimage.data.chelsea()
    rects = main(image)
    arr = np.array(rects)
    print(arr)
    # images_path = 'E:\\Snapshot Serengeti\\data\\img_multi'
    # filesList = os.listdir(images_path)
    # for fileName in filesList:
    #     fileAbsPath = os.path.join(images_path, fileName)
    #     image = imread(fileAbsPath)  # 返回 ndarray
    #     rects = main(image)
    #     arr = np.array(rects)
    #     print(arr)
