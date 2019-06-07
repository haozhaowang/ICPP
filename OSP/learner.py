# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from threading import Thread, Lock
from multiprocessing import Value, Queue
from multiprocessing.managers import BaseManager
from ctypes import c_bool, c_float, c_int

import torch
import torch.distributed.deprecated as dist
from cjltest.divide_data import partition_dataset, select_dataset
from cjltest.models import MnistCNN, AlexNetForCIFAR
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import MySGD, test_model
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets,models, transforms
import ResNetOnCifar10

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=1)
parser.add_argument('--workers-num', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='MnistCNN')
parser.add_argument('--save-path', type=str, default='./')

# 参数信息
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--train-bsz', type=int, default=200)
parser.add_argument('--stale-threshold', type=int, default=100)

args = parser.parse_args()


# send local update to server
# send: local_update, it_count
def sender(model_cache, global_update, local_update, it_count,
           loss_t, update_lock, data_lock, group, receive_end,
           batch_communication_interval, stale_in_iteration):
    comm_count = 0
    while True:
        # this is a queue that is controled by computer module
        # At lease one gradient has been generated, the sender has gradient to send
        update_lock.get()

        # Note: A lock should be here before accessing to local_update
        # copy local update to global update, and then set local update to be 0
        data_lock.acquire()
        loss = loss_t.data
        loss_t.data = torch.tensor(0.)
        it_times = it_count.value
        it_count.value = 0.
        for idx, update in enumerate(global_update):
            update.data = local_update[idx].data
            local_update[idx].data = torch.zeros_like(update.data)
        data_lock.release()

        comm_s = time.time()
        loss_it = torch.tensor([float(loss), it_times])
        dist.gather(tensor=loss_it, dst=0, group=group)
        for idx, update in enumerate(global_update):
            dist.gather(tensor=update.data, dst=0, group=group)

        for idx, param in enumerate(model_cache):
            dist.scatter(tensor=param.data, src=0, group=group)
        receive_end.value = True

        comm_e = time.time()
        comm_count += 1
        # compute average communication time of an iteration
        batch_communication_interval.value = (batch_communication_interval.value * (comm_count-1) +
                                              (comm_e - comm_s))/comm_count
        stale_in_iteration.value = (stale_in_iteration.value * (comm_count-1) + it_times)/comm_count
    return

# noinspection PyTypeChecker
def run(rank, workers, model, save_path, train_data, test_data, global_lr):
    # 获取ps端传来的模型初始参数
    print(workers)

    _group = [w for w in workers].append(0)
    group = dist.new_group(_group)

    for p in model.parameters():
        tmp_p = torch.zeros_like(p)
        dist.scatter(tensor=tmp_p, src=0, group=group)
        p.data = tmp_p
    print('Model recved successfully!')

    temp_lr = global_lr.get()

    if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10']:
        optimizer = MySGD(model.parameters(), lr=temp_lr)
    else:
        optimizer = MySGD(model.parameters(), lr=temp_lr)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print('Begin!')

    # the parameters that will be transferred to the thread
    model_cache = [p.data+0.0 for p in model.parameters()]
    global_update = [torch.zeros_like(p) for p in model.parameters()]
    local_update = [torch.zeros_like(p) for p in model.parameters()]
    it_count = Value(c_float, 0.)    # count update times in an iteration by local worker
    data_lock = Lock()
    update_lock = Queue()
    update_lock.put(1)

    loss_t = torch.tensor(0.0)
    receive_end = Value(c_bool, False)
    batch_communication_interval = Value(c_float, 0.0)
    stale_in_iteration = Value(c_float, 0.)

    sender_td = Thread(target=sender, args=(model_cache, global_update,
                                            local_update, it_count,
                                            loss_t, update_lock,
                                            data_lock, group,
                                            receive_end,
                                            batch_communication_interval,
                                            stale_in_iteration,),
                       daemon=True)
    sender_td.start()

    time_logs = open("./record"+str(rank), 'w')
    osp_logs = open("./log" + str(rank), 'w')
    Stale_Threshold = args.stale_threshold
    for epoch in range(args.epochs):
        batch_interval = 0.0
        batch_comp_interval = 0.0
        s_time = time.time()
        model.train()

        # AlexNet在指定epoch减少学习率LR
        # learning rate should be decreased on server due to unmatched updating speed between local worker and server
        if not global_lr.empty():
            g_lr = global_lr.get()
            if args.model == 'AlexNet':
                for param_group in optimizer.param_groups:
                    param_group['lr'] = g_lr
                    print('LR Decreased! Now: {}'.format(param_group['lr']))

        for batch_idx, (data, target) in enumerate(train_data):
            batch_start_time = time.time()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizer.get_delta_w()

            optimizer.step()

            # Aggregate local update
            data_lock.acquire()
            # aggregate loss
            loss_t.data += loss.data
            it_count.value += 1
            for g_idx, update in enumerate(local_update):
                update.data += delta_ws[g_idx].data
            data_lock.release()

            batch_computation_time = time.time()

            # Open the lock once the local update has at least one gradient
            if it_count.value == 1:
                update_lock.put(1)
            while it_count.value >= Stale_Threshold:
                pass

            if receive_end.value:
                receive_end.value = False
                for idx, param in enumerate(model.parameters()):
                    param.data = model_cache[idx]       # without local update
                    # param.data = model_cache[idx] - global_update[idx] # with local update

            batch_end_time = time.time()
            batch_interval += batch_end_time - batch_start_time
            batch_comp_interval += batch_computation_time - batch_start_time
            osp_logs.write(str(batch_end_time - batch_start_time)+ "\t"
                           + str(batch_computation_time - batch_start_time)
                           +"\n")
            osp_logs.flush()

        print('Rank {}, Epoch {}, Loss:{}'
             .format(rank, epoch, loss.data.item()))

        e_time = time.time()
        # 训练结束后进行test
        #test_loss, acc = test_model(rank, model, test_data, criterion=criterion)
        acc = 0.0
        batch_interval /= batch_idx
        batch_comp_interval /= batch_idx
        logs = torch.tensor([acc, batch_interval, batch_comp_interval, batch_communication_interval.value, stale_in_iteration.value])
        time_logs.write(str(logs)+'\n')
        time_logs.flush()
        # dist.gather(tensor=logs, dst = 0, group = group)
    time_logs.close()
    sender_td.join()



def init_processes(rank, size, workers,
                   model, save_path,
                   train_dataset, test_dataset,
                   global_lr,
                   fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, workers, model, save_path, train_dataset, test_dataset, global_lr)


if __name__ == '__main__':

    workers = [v+1 for v in range(args.workers_num)]

    if args.model == 'MnistCNN':
        model = MnistCNN()
        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                                      transform=test_transform)
    elif args.model == 'AlexNet':
        train_transform, test_transform = get_data_transform('cifar')

        if args.data_name == 'cifar10':
            model = AlexNetForCIFAR()
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                            transform=test_transform)
        else:
            model = AlexNetForCIFAR(num_classes=100)
            train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=True,
                                              transform=train_transform)
            test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=True,
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
    train_bsz /= len(workers)
    train_bsz = int(train_bsz)

    train_data = partition_dataset(train_dataset, workers)
    test_data = partition_dataset(test_dataset, workers)

    this_rank = args.this_rank
    train_data = select_dataset(workers, this_rank, train_data, batch_size=train_bsz)
    test_data = select_dataset(workers, this_rank, test_data, batch_size=test_bsz)

    # 用所有的测试数据测试
    #test_data = DataLoader(test_dataset, batch_size=test_bsz, shuffle=True)

    world_size = len(workers) + 1

    save_path = str(args.save_path)
    save_path = save_path.rstrip('/')


    class LrManager(BaseManager): pass
    LrManager.register('get_global_lr')
    manager = LrManager(address=(args.ps_ip, 5000), authkey=b'overlap')
    manager.connect()
    global_lr = manager.get_global_lr()

    p = TorchProcess(target=init_processes, args=(this_rank, world_size,
                                                  workers, model, save_path,
                                                  train_data, test_data, global_lr,
                                                  run))
    p.start()
    p.join()
