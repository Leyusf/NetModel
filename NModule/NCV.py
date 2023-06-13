import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision


def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), dim=-1)
    return boxes


def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes


def corner_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def open_img_and_box_demo():
    butterfly_bbox, flower_bbox = [50.0, 25.0, 175.0, 175.0], [175.0, 100.0, 250.0, 175.0]
    img = plt.imread("..\\data\\images\\img.png")
    fig = plt.imshow(img)
    fig.axes.add_patch(corner_to_rect(butterfly_bbox, 'blue'))
    fig.axes.add_patch(corner_to_rect(flower_bbox, 'red'))
    plt.show()


def read_dataset(data_dir, is_train=True, edge_size=256):
    """读取目标检测数据集中的图像和标签"""
    csv_fname = os.path.join(data_dir, 'train' if is_train
    else 'valid', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'train' if is_train else
            'valid', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / edge_size



