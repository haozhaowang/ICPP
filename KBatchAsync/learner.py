# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
import time
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager

import numpy as np
import torch
import torch.distributed.deprecated as dist
from cjltest.divide_data import partition_dataset, select_dataset, DataPartitioner
from cjltest.models import MnistCNN, AlexNetForCIFAR
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import MySGD, test_model
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torchvision import datasets, models, transforms
import ResNetOnCifar10

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=1)
parser.add_argument('--workers-num', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--save-path', type=str, default='./')
parser.add_argument('--model', type=str, default='MnistCNN')

# 参数信息
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--train-bsz', type=int, default=200)
parser.add_argument('--test-bsz', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--stale-threshold', type=int, default=1000)    # Controls the number of minibatch


args = parser.parse_args()


def run(rank, model, train_data, test_data, queue, param_q, stop_flag):
    # 获取ps端传来的模型初始参数
    while True:
        if not param_q.empty():
            param_dict = param_q.get()
            tmp = OrderedDict(map(lambda item: (item[0], torch.from_numpy(item[1])),
                                  param_dict.items()))
            model.load_state_dict(tmp)
            break
    print('Model recved successfully!')

   
    if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10']:
        optimizer = MySGD(model.parameters(), lr=0.1)
    else:
        optimizer = MySGD(model.parameters(), lr=0.01)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.model in ['AlexNet', 'ResNet18OnCifar10']:
        decay_period = 50
    else:
        decay_period = 100
    print('Begin!')

    time_logs = open("./record" + str(rank), 'w')
    for epoch in range(int(args.epochs)):
        batch_interval = 0.0
        batch_comp_interval = 0.0
        batch_comm_interval = 0.0
        batch_push_interval = 0.0
        batch_pull_interval = 0.0
        model.train()
        # AlexNet在指定epoch减少学习率LR
        #if args.model == 'AlexNet':
        if (epoch+1) % decay_period == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                print('LR Decreased! Now: {}'.format(param_group['lr']))
        epoch_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_data):
            batch_start_time = time.time()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizer.get_delta_w()

            batch_comp_time = time.time()
            # noinspection PyBroadException
            try:  # 捕获异常，异常来源于ps进程的停止
                if delta_ws:
                    queue.put({
                        rank: [loss.data.numpy(), np.array(args.train_bsz), False]
                    })
                for delta in delta_ws:
                    dist.send(tensor=delta, dst=0)

                batch_push_time = time.time()

                for idx, param in enumerate(model.parameters()):
                    tmp_tensor = torch.zeros_like(param.data)
                    dist.recv(tensor= tmp_tensor, src=0)
                    param.data = tmp_tensor

                batch_tmp_time = time.time()
                batch_pull_time = time.time()
                #print('Rank {}, Epoch {}, Batch {}/{}, Loss:{}'
                #     .format(rank, epoch, batch_idx, len(train_data), loss.data[0]))
            except Exception as e:
                print(str(e))
                print('Should Stop: {}!'.format(stop_flag.value))
                break

            batch_interval += batch_pull_time - batch_start_time
            batch_comp_interval += batch_comp_time - batch_start_time
            batch_comm_interval += batch_pull_time - batch_comp_time
            batch_push_interval += batch_push_time - batch_comp_time
            batch_pull_interval += batch_pull_time - batch_push_time
            b_interval = batch_interval / (batch_idx+1)
            b_comp_interval = batch_comp_interval /(batch_idx+1)
            b_comm_interval = batch_comm_interval / (batch_idx+1)
            b_push_interval = batch_push_interval / (batch_idx+1)
            b_pull_interval = batch_pull_interval / (batch_idx+1)
            logs = torch.tensor([0.0, b_interval, b_comp_interval, b_comm_interval, b_push_interval,
                                 b_pull_interval, batch_pull_time - batch_tmp_time])
            time_logs.write(str(logs) + '\n')
            time_logs.flush()

        batch_interval /= batch_idx
        batch_comp_interval /= batch_idx
        batch_comm_interval /= batch_idx
        batch_push_interval /= batch_idx
        batch_pull_interval /= batch_idx
        logs = torch.tensor([0.0, batch_interval, batch_comp_interval, batch_comm_interval, batch_push_interval, batch_pull_interval])
        time_logs.write(str(epoch) + '\t' + str(logs) + '\n')
        time_logs.flush()
        # 训练结束后进行test
        print("test Model:",epoch)
        # test_model(rank, model, test_data, criterion=criterion)
        if stop_flag.value:
            break
    queue.put({rank: [[], [], True]})
    time_logs.close()
    print("Worker {} has completed epoch {}!".format(args.this_rank, epoch))

def init_processes(rank, size, model,
                   train_dataset, test_dataset,
                   q, param_q, stop_flag,
                   fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, model, train_dataset, test_dataset, q, param_q, stop_flag)


def capture_stop(stop_signal, flag: Value):
    while True:
        if not stop_signal.empty():
            flag.value = True
            print('Time Up! Stop: {}!'.format(flag.value))
            break


if __name__ == "__main__":

    """
    判断使用的模型，MnistCNN或者是AlexNet
    模型不同，数据集、数据集处理方式、优化函数、损失函数、参数等都不一样
    """
    if args.model == 'MnistCNN':
        model = MnistCNN()
        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=test_transform)
    elif args.model == 'AlexNet':
        model = AlexNetForCIFAR()

        train_transform, test_transform = get_data_transform('cifar')

        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                        transform=test_transform)
    elif args.model == 'LROnMnist':
        model = ResNetOnCifar10.LROnMnist()
        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=test_transform)
    elif args.model == 'LROnCifar10':
        model = ResNetOnCifar10.LROnCifar10()
        train_transform, test_transform = get_data_transform('cifar')

        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                       transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                      transform=test_transform)
    elif args.model == 'ResNet18OnCifar10':
        model = ResNetOnCifar10.ResNet18()

        train_transform, test_transform = get_data_transform('cifar')
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                        transform=test_transform)
    elif args.model == 'ResNet34':
        model = models.resnet34(pretrained=False)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_transform = train_transform
        train_dataset = datasets.ImageFolder(args.data_dir, train=True, download=False,
                                         transform=train_transform)
        test_dataset = datasets.ImageFolder(args.data_dir, train=False, download=False,
                                        transform=test_transform)
    else:
        print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
        sys.exit(-1)

    train_bsz = args.train_bsz
    test_bsz = 200
    workers = [v+1 for v in range(args.workers_num)]
    train_data = partition_dataset(train_dataset, workers)
    test_data = partition_dataset(test_dataset, workers)

    this_rank = args.this_rank
    train_data = select_dataset(workers, this_rank, train_data, batch_size=train_bsz)
    test_data = select_dataset(workers, this_rank, test_data, batch_size=test_bsz)

    world_size = len(workers) + 1


    class MyManager(BaseManager):
        pass


    MyManager.register('get_queue')
    MyManager.register('get_param')
    MyManager.register('get_stop_signal')
    manager = MyManager(address=(args.ps_ip, 5000), authkey=b'queue')
    manager.connect()

    q = manager.get_queue()  # 更新参数使用的队列
    param_q = manager.get_param()  # 接收初始模型参数使用的队列
    stop_signal = manager.get_stop_signal()  # 接收停止信号使用的队列

    stop_flag = Value(c_bool, False)
    # 开启一个进程捕获ps的stop信息
    stop_p = Process(target=capture_stop,
                     args=(stop_signal, stop_flag))

    p = TorchProcess(target=init_processes, args=(this_rank, world_size,
                                                  model,
                                                  train_data, test_data,
                                                  q, param_q, stop_flag,
                                                  run))
    p.start()
    stop_p.start()
    p.join()
    stop_p.join()