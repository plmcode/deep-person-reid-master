from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from optimizers import init_optim

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory") #数据集地址
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())   #所使用的数据集
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")   #dataloader（）线程数 这儿是4线程，读取数据
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options    训练的信息
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")  #选择的优化器
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")      #从哪开始训练，0是从头开始训练，假如上次在哪停止，下次从这儿开始训练，
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")   #训练的bath大小
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")   #测试的bath大小
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=20, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")   #学习阈，通常训练多少个epho，就会下降 即训练20个就会下降一次
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")    #具体下降多少，在这儿就是0.1
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")  #模型正则化参数，使得模型复杂度成为loss的一部分，防止过拟合
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())  #选择的模型
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")   #打印的频率，多少次就会打印一次log
parser.add_argument('--seed', type=int, default=1, help="manual seed")   #控制一些随即参数，保证结果可以稳定复现
parser.add_argument('--resume', type=str, default='', metavar='PATH')   #从那个模型开始恢复，就是前面中断训练，恢复时需指定
parser.add_argument('--evaluate', action='store_true', help="evaluation only") #指定是在做训练还是测试，默认关闭，如果打开，则只做测试而不训练
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)") #做多少个epho才做一次测试，默认-1，即训练结束后才做测试
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch") #从多少个epho做测试
parser.add_argument('--save-dir', type=str, default='log')  #存放log、train point的路径
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices   #用前面参数设定的第参数块gou训练
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:  #如果不是测试模式，则
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))  #当前的gpu
        cudnn.benchmark = True   #cudnn.benchmark 使用cudnn库，使得卷积训练的速度加快
        torch.cuda.manual_seed_all(args.seed)  #保证结果可以稳定的复现
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))   #数据集初始化
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )   #后面的参数是针对chk数据集的

#由于训练和测试对图片的要求不同，训练时需要对图片进行增广，测试时，需要的式原图片，故需要两个函数

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),   #先放大再裁剪
        T.RandomHorizontalFlip(),   #水平反转 有概率反转
        T.ToTensor(),  #将其转化为Tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #图像色素做规划，三个通道的，其小数是固定的
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)), #不需要做增广，但是需要将其化为同样的尺寸
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False    #节约显存

#有三个数据集，故需要写三个dataloader（）
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch, shuffle=True,   #将读取顺序打乱
        num_workers=args.workers, #线程数
        pin_memory=pin_memory,   #内存数
        drop_last=True,  # 是否丢弃尾部的数据 如100张图片，batchsize=32 训练4个epho 用了96 剩余的四张可以丢弃
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))  #输出所使用的模型
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0)) #模型的大小

    #分类损失
    #criterion = nn.CrossEntropyLoss()  项目中作者进行了重构

    criterion = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    #优化器 作者进行了重构 原来是
    #optimizer = torch.optim.Adam()
    optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
    '''
       model.parameters()表示优化器会更新模型中的所有参数 假如·更新某一层的参数吗，则model.fc 更新fc层的参数
       再者就是同时更新两层，用   
       nn.Sequential([ 
        model.conv1,
        model.cov2
    ])
    args.lr 表示初始学习率 前面定义的
    args.weight_decay 模型的正则化参数
    '''
    if args.stepsize > 0:    #学习率衰减
        '''
         学习率衰减的作用，跟损失函数挂钩，学习率也就是步长，在寻找损失函数最小值时，通过学习率衰减，逐步减小步长，进行寻找
        '''
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch  #从那开始训练

    if args.resume: #是否恢复模型
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:  #将模型包装成并行
        model = nn.DataParallel(model).cuda()

    if args.evaluate:  #做测试
        print("Evaluate only")
        test(model, queryloader, galleryloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):   #开始训练
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        
        if args.stepsize > 0: scheduler.step()   #学习率衰减
        
        if (epoch+1) > args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test") #经过多少轮后会进行测试
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,    #模型的参数
                'rank1': rank1,     #当前模型的准确度
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    losses = AverageMeter()   #loss的时间
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train() #处于训练阶段

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):  #每次迭代会读取相应的数据
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()   #将图像放到gpu中

        # measure data loading time
        data_time.update(time.time() - end)
        
        outputs = model(imgs)  #输出分类的标签  前传
        if isinstance(outputs, tuple):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)    #计算损失
        optimizer.zero_grad()  #然后将optimizer 梯度清空
        loss.backward()  #反传
        optimizer.step()  #更新模型参数

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):   #测试脚本
    batch_time = AverageMeter()
    
    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]

if __name__ == '__main__':
    main()