from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import numpy as np

class Random2DTranslation(object):   #将图片进行resizw，即将图片扩大1/8，并且随机裁剪一个区域
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5. 概率，即有多大概率执行裁剪 默认是0.5
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR): #参数介绍：resize 后的宽度和高度 概率 以及涉及裁剪，裁剪后的差值 默认双线性差值
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):  #一张图片中想要的区域进行达到要求的大小 一般两种做法，一种直接扩大，然后裁剪，另一种是先选取想要的部分，然后进行扩大
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p: #就是产生随机数，当小于概率时，不进行增广
            return img.resize((self.width, self.height), self.interpolation) #注意PIL是先宽度后高度 cv是先高度后宽度
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125)) #round是取上整   1.125是扩大9/8的大小
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width       #裁剪的大小
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))   #产生随机数，即裁剪的开始位置
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))   #裁剪区域
        return croped_img

if __name__ == '__main__':
    pass