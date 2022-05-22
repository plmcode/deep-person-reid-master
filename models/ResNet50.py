from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from IPython import embed


class ResNet50(nn.Module):  #写网络的时候 不许继承pytorch库
    def __init__(self, num_classes, loss={'xent'}, **kwargs):  #num_class 与reid挂钩  loss={} 损失函数 其中有softmax。metric等，loss的不同区分表征学习和度量学习
        super(ResNet50, self).__init__()   #针对python2继承的代码，如果是python3则需要更改
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)   #分割、训练等 用pretrained进行预加载
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])  # 经过平均池化层进行处理 池化层的输入是图像的大小 x.size()[2:] = （8，4）
        f = x.view(x.size(0), -1)
        if not self.training:    #测试和训练时，进行区分 其中y是做分类损失的，f是来做检索的 测试时时是不关心y的
            return f
        y = self.classifier(f)
        return y


if __name__ == '__main__':
    model = ResNet50(num_classes=751)
    imgs = torch.Tensor(32, 3, 256, 128)
    f  = model.forward(imgs)
    embed()
    #f = model(imgs)
