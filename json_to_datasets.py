# -*- coding: utf-8 -*-
# @Time    : 2021-05-25 15:42
# @Author  : Wu You
# @File    : json_to_datasets.py
# @Software: PyCharm


import base64
import argparse
import json
import os
import warnings
import cv2
import PIL.Image
import yaml
import time
from multiprocessing import Pool
import functools
from labelme import utils


def main(json_file, img_path, mask_path, yaml_path):
    data = json.load(open(json_file))

    if data['imageData']:
        imageData = data['imageData']
    else:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    # img = cv2.imread(os.path.join(img_dir, os.path.basename(json_file).split('.')[0] + '.jpg'))
    label_name_to_value = {'_background_': 0}
    for shape in data['shapes']:
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value

    # label_values 必须是密集的
    label_values, label_names = [], []
    for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
        label_values.append(lv)
        label_names.append(ln)
    assert label_values == list(range(len(label_values)))

    print(label_name_to_value)
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    captions = ['{}: {}'.format(lv, ln)
                for ln, lv in label_name_to_value.items()]
    lbl_viz = utils.draw_label(lbl, img, captions)

    # 保存原图
    PIL.Image.fromarray(img).save(os.path.join(img_path, os.path.basename(json_file).split('.')[0] + '.jpg'))
    # 保存二值图
    utils.lblsave(os.path.join(mask_path, os.path.basename(json_file).split('.')[0] + '.png'), lbl)
    PIL.Image.fromarray(lbl_viz).save(os.path.join(viz_path, os.path.basename(json_file).split('.')[0] + '.png'))
    # 保存txt文件
    with open(os.path.join(txt_path, os.path.basename(json_file).split('.')[0] + '.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')

    info = dict(label_names=label_names)
    # 保存yaml文件
    with open(os.path.join(yaml_path, os.path.basename(json_file).split('.')[0] + '.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)

    print('Saved to: %s' % os.path.join(mask_path, os.path.basename(json_file).split('.')[0] + '.png'))


if __name__ == '__main__':
    start = time.time()
    # 图片和json文件路径以及保存文件路径
    img_dir = 'E:/Robot_video_detection/ori_img'
    json_path = "C:/Users/admin/Desktop/json"
    save_dir = 'C:/Users/admin/Desktop'
    img_path = save_dir + '/img'
    mask_path = save_dir + '/mask'
    yaml_path = save_dir + '/yaml'
    viz_path = save_dir + '/viz'
    txt_path = save_dir + '/txt'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if not os.path.exists(yaml_path):
        os.makedirs(yaml_path)
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    json_file = []
    for name in os.listdir(json_path):
        json_file.append(os.path.join(json_path, name))
    # json_file = glob.glob(os.path.join(json_path, '*.json'))

    newfunc = functools.partial(main, img_path=img_path, mask_path=mask_path, yaml_path=yaml_path)

    pool = Pool(16)
    pool.map(newfunc, json_file)
    # pool.map(main, json_file)
    pool.close()
    pool.join()
    end = time.time()
    print(end - start)
