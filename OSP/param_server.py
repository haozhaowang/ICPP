# -*- coding: utf-8 -*-
import argparse
import math
import os
import random
import sys
import time

from ctypes import c_float
from multiprocessing import Value, Queue
from multiprocessing.managers import BaseManager

import torch
import torch.distributed.deprecated as dist
from cjltest.models import MnistCNN, AlexNetForCIFAR
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import test_model
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import ResNetOnCifar10

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=0)
parser.add_argument('--workers-num', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/data')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='MnistCNN')

# 参数信息
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--train-bsz', type=int, default=200)
parser.add_argument('--data-ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.1)

args = parser.parse_args()

# noinspection PyTypeChecker
def run(rank, model, train_pics, train_bsz, g_lr):
    workers = [v+1 for v in range(args.workers_num)]
    print(workers)

    _group = [w for w in workers].append(rank)
    group = dist.new_group(_group)

    for p in model.parameters():
        scatter_p_list = [p.data for _ in range(len(workers) + 1)]
        dist.scatter(tensor=p.data, scatter_list=scatter_p_list, group=group)

    # initialize learning rate

    if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10']:
        global_lr = 0.1
    else:
        global_lr = 0.01

    for w in workers:
        g_lr.put(global_lr)
    print('Model Sent Finished!')

    print('Begin!')

    epoch_train_loss = 0.0
    trainloss_file = './trainloss' + args.model + '.txt'
    log_file = './log' + args.model + '.txt'
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    if (os.path.isfile(log_file)):
        os.remove(log_file)
    f_trainloss = open(trainloss_file, 'a')
    f_log = open(log_file, 'a')

    tmp = [(0, 0) for _ in range(int(math.ceil(int(train_pics * args.data_ratio) / (len(workers) * train_bsz))))]

    s_time = time.time()
    epoch_time = s_time
    total_iteration_time = 0
    iteration_times_count_epoch = 0
    iteration_times_epoch = len(tmp) * len(workers)
    if args.model in ['AlexNet', 'ResNet18OnCifar10']:
        decay_period = 50
    else:
        decay_period = 100

    global_clock = 0
    epoch_clock = 0
    real_epoch = 0
    for epoch in range(args.epochs):
        for batch_idx, (_, _) in enumerate(tmp):
            batch_start_time = time.time()

            # receive the list of train loss and local iteration count from workers
            loss_it_list = [torch.tensor([0.0, 0.0]) for _ in range(len(workers) + 1)]
            dist.gather(tensor=torch.tensor([0.0, 1.0]), gather_list=loss_it_list, group=group)

            loss_avg = [loss_it[0]/loss_it[1] for loss_it in loss_it_list]
            epoch_train_loss += sum(loss_avg).item()/len(workers)
            iteration_times = sum(loss_it_list)[1]
            total_iteration_time += iteration_times
            iteration_times_count_epoch += iteration_times

            # receive global update from each worker
            for update_idx, param in enumerate(model.parameters()):
                tensor = torch.zeros_like(param.data)
                # FIXME FIXED：gather_list中的每个Tensor都必须是新的对象，否则会出问题
                gather_list = [torch.zeros_like(param.data) for _ in range(len(workers) + 1)]
                dist.gather(tensor=tensor, gather_list=gather_list, group=group)
                # here we only use average temperally for simplicity
                tensor = sum(gather_list) / len(workers)
                param.data -= tensor
            # send updated model back to workers
            for param_idx, param in enumerate(model.parameters()):
                scatter_list = [param.data for _ in range(len(workers) + 1)]
                dist.scatter(tensor=torch.zeros_like(param.data), scatter_list=scatter_list, group=group)

            global_clock += 1
            epoch_clock += 1
            # AlexNet在指定epoch减少学习率LR
            # not update intra the epoch
            temp_epoch = int(total_iteration_time / iteration_times_epoch) + 1
            if temp_epoch > real_epoch:
                real_epoch = temp_epoch
                if real_epoch % decay_period == 0:
                    global_lr *= 0.1
                    for w in workers:
                        g_lr.put(global_lr)
                    print('LR Decreased! Now: {}'.format(global_lr))

            # evaluate the time of each iteration
            batch_end_time = time.time()
            batch_time_interval = batch_end_time - batch_start_time
            f_log.write(str(batch_time_interval) + "\t" + str(iteration_times)+ "\n")
            f_log.flush()


            if iteration_times_count_epoch >= iteration_times_epoch:
                e_epoch_time = time.time()
                iteration_times_count_epoch -= iteration_times_epoch
                # test_acc, batch_interval, batch_comp_interval, batch_comm_interval
                logs = torch.tensor([0.0, 0.0, 0.0, 0.0])
                logs_list = [torch.zeros_like(logs) for _ in range(len(workers) + 1)]
                # dist.gather(tensor=logs, gather_list=logs_list, group=group)
                test_acc, batch_interval, batch_comp_interval, batch_comm_interval = zip(*logs_list)
                test_acc = sum(test_acc) / len(workers)
                batch_interval = sum(batch_interval) / len(workers)
                batch_comp_interval = sum(batch_comp_interval) / len(workers)
                batch_comm_interval = sum(batch_comm_interval) / len(workers)

                f_trainloss.write(str(args.this_rank) +
                                  "\t" + str(epoch_train_loss / float(epoch_clock)) +         ###### This place has problem.
                                  "\t" + str(0) +
                                  "\t" + str(e_epoch_time - epoch_time) +
                                  "\t" + str(e_epoch_time - s_time) +
                                  "\t" + str(int(total_iteration_time / iteration_times_epoch)) +
                                  "\t" + str(test_acc.item()) +
                                  "\t" + str(batch_interval.item()) +
                                  "\t" + str(batch_comp_interval.item()) +
                                  "\t" + str(batch_comm_interval.item()) +
                                  "\t" + str(global_clock) +
                                  '\n')
                print("total_iteration_time:{}, iteration_times:{}".format(total_iteration_time, iteration_times_epoch))
                f_trainloss.flush()
                epoch_time = e_epoch_time
                epoch_train_loss = 0.0
                epoch_clock = 0

        print('Done Epoch {}/{}!'.format(epoch + 1, args.epochs))

    f_trainloss.close()


def init_processes(rank, size,
                   model, train_pics, train_bsz, g_lr,
                   fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, model, train_pics, train_bsz, g_lr)


if __name__ == '__main__':
    # 随机数设置
    manual_seed = 1
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    workers = [v+1 for v in range(args.workers_num)]

    if args.model == 'MnistCNN':
        model = MnistCNN()
        train_pics = 60000
    elif args.model == 'AlexNet':
        if args.data_name == 'cifar10':
            model = AlexNetForCIFAR()
        else:
            model = AlexNetForCIFAR(num_classes=100)
        train_pics = 50000
    elif args.model == 'LROnMnist':
        model = ResNetOnCifar10.LROnMnist()
        train_transform, test_transform = get_data_transform('mnist')
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=test_transform)
        train_pics = 60000
    elif args.model == 'LROnCifar10':
        model = ResNetOnCifar10.LROnCifar10()
        train_transform, test_transform = get_data_transform('cifar')
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                      transform=test_transform)
        train_pics = 50000
    elif args.model == 'ResNet18OnCifar10':
        model = ResNetOnCifar10.ResNet18()

        train_transform, test_transform = get_data_transform('cifar')
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                        transform=test_transform)
        train_pics = 50000
    elif args.model == 'ResNet34':
        model = models.resnet34(pretrained=False)

        test_transform, train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        test_dataset = datasets.ImageFolder(args.data_dir, train=False, download=False,
                                        transform=test_transform)
        train_pics = 121187
    else:
        print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
        sys.exit(-1)

    train_bsz = args.train_bsz
    train_bsz /= len(workers)

    world_size = args.workers_num + 1
    this_rank = args.this_rank

    global_lr = Queue()
    class LrManager(BaseManager): pass
    LrManager.register('get_global_lr', callable=lambda : global_lr)
    manager = LrManager(address=(args.ps_ip, 5000), authkey = b'overlap')
    manager.start()

    g_lr = manager.get_global_lr()

    p = TorchProcess(target=init_processes, args=(this_rank, world_size,
                                                  model, train_pics,
                                                  train_bsz, g_lr,
                                                  run))
    p.start()
    p.join()
    manager.shutdown()