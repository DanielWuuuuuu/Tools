"""
将正样本的xml转为txt
"""
import sys
import codecs
import os
import xml.etree.ElementTree as ET

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

xml_dir = 'D:/project/syndata-generation-master/data_generator/origin_with_none2_random_lc/y_pistol/xml/'  # xml文件目录
txt_dir = 'D:/project/syndata-generation-master/data_generator/origin_with_none2_random_lc/y_pistol/txt/'  # 保存txt的路径
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

# 类别信息
class_number = 11
class_name_0 = ["person", "0"]
class_name_1 = ["table", "1"]
class_name_2 = ["chair", "2"]
class_name_3 = ["backpack", "bag", "3"]
class_name_4 = ["suitcase", "4"]
class_name_5 = ["closet", "5"]
class_name_6 = ["gun", "6"]
class_name_7 = ["stairs", "7"]
class_name_8 = ["car", "8"]
class_name_9 = ["door", "9"]
class_name_10 = ["column", "10"]
class_all = [class_name_0, class_name_1, class_name_2, class_name_3, class_name_4,
             class_name_5, class_name_6, class_name_7, class_name_8, class_name_9,
             class_name_10]

# 统计各类别的数量
count = [0] * class_number

for fp in os.listdir(xml_dir):
    try:
        root = ET.parse(os.path.join(xml_dir, fp)).getroot()
    except:
        pass
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz[0].text)
    height = float(sz[1].text)
    filename = root.find('filename').text

    # 找到图片中的所有框
    for child in root.findall('object'):
        # 找到类别名
        name = child.find('name')
        name = name.text
        flag_name = False
        for i in range(class_number):
            if name in class_all[i]:
                name = str(i)
                flag_name = True
                count[i] += 1
                break
        if not flag_name:
            print("给定类别中无此类别：" + name)

        # 找到框的标注值并进行读取
        sub = child.find('bndbox')
        xmin = float(sub[0].text)
        ymin = float(sub[1].text)
        xmax = float(sub[2].text)
        ymax = float(sub[3].text)
        # 转换成yolov3的标签格式（归一化到[0-1]）
        try:
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
        except ZeroDivisionError:
            print(filename, '的 width有问题')

        # 写入txt
        with open(os.path.join(txt_dir, fp.split('.')[0] + '.txt'), 'a+') as f:
            f.write(' '.join([str(name), str(x_center), str(y_center), str(w), str(h) + '\n']))

for i in range(class_number):
    print(i, "类共有" + str(count[i]) + "份标注")
