from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F  # 主要是写平均池化层
import torchvision

# ResNet 有50层，可以调用pytorch进行实现
__all__ = ['ResNet50', 'ResNet101', 'ResNet50M']


class ResNet50(nn.Module):  # 写网络的时候 不许继承pytorch库
    def __init__(self, num_classes, loss={'xent'},
                 **kwargs):  # num_class 与reid挂钩  loss={} 损失函数 其中有softmax。metric等，loss的不同区分表征学习和度量学习
        super(ResNet50, self).__init__()  # 针对python2继承的代码，如果是python3则需要更改
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)  # 分割、训练等 用pretrained进行预加载
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        '''
        Resnet50 最后两层即一个平均池化层和分类层，池化层输入图像大小为7，与图像大小有关，最后一层分类层，原先是输入2048维度，输出是1000维度，需要对其进行修改  
        这与我们输入的id数有关,renst50.childern()是一个迭代器,输出结果为resnet50的结构，[:-2]就是取前面除了最后两层的结果
        nn.Sequential(*list(resnet50.children())[:-2])将指针变为网络
        '''
        self.classifier = nn.Linear(2048, num_classes)  # 分类器，输入2048，输出为我们的reid数即751
        self.feat_dim = 2048  # feature dimension

    def forward(self, x):  # 前传函数，
        x = self.base(x)  # 源数据(32, 3, 256, 128)经过前向传播后 样本 通道 大小 torch.Size([32, 2048, 8, 4])
        x = F.avg_pool2d(x, x.size()[2:])  # 经过平均池化层进行处理 池化层的输入是图像的大小 x.size()[2:] = （8，4）
        f = x.view(x.size(0), -1)
        # 一般不对特征做规划，若做规划化，规划就是假如对a做规划，就是a/||a||
        #f = 1. * f / (torch.norm(f, 2, -1, keepdim=True).expand_as(f) + 1e-12)
        # 1.0*f转为浮点数
        # -1表示最后一维，这儿由于f是二维的，所以-1和1都可以 最后的添加一个小数，因为其在分母上，不能为0
        '''
        如果直接进行分类，因为x在经过池化层后，输出为（32, 2048, 1, 1），而torch.Tensor(32,2048),所以要进行处理转化为（(32,2048)），将后面的尺寸去掉
        x.view() 就是对tensor进行reshape操作
                import torch
                v1 = torch.range(1, 4)
                v2 = v1.view(2, 2)
                print(v2)
                v3 = v2.view(4,-1)
                print(v3)
                输出：
                2行2列
                4行1列
        view()函数的功能根reshape类似，用来转换size大小。x = x.view(batchsize, -1)中batchsize指转换后有几行，
        而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        '''
        if not self.training:  # 测试和训练时，进行区分 其中y是做分类损失的，f是来做检索的 测试时时是不关心y的
            return f
        y = self.classifier(f)  # 进行分两类，然后输出（32，751） 那么多图片将其分类为751类

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048  # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """

    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.feat_dim = 3072  # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)

        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
