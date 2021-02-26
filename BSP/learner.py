# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import torch
import torch.distributed.deprecated as dist
from cjltest.divide_data import partition_dataset, select_dataset
from cjltest.models import MnistCNN, AlexNetForCIFAR
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import MySGD, test_model
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import ResNetOnCifar10

parser = argparse.ArgumentParser()
# Information of the cluster
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=1)
parser.add_argument('--workers-num', type=int, default=2)

# Models and Dataset
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='MnistCNN')
parser.add_argument('--save-path', type=str, default='./')

# Hyperparameters for the algorithm
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--train-bsz', type=int, default=200)
parser.add_argument('--stale-threshold', type=int, default=0)

args = parser.parse_args()


# noinspection PyTypeChecker
def run(rank, workers, model, save_path, train_data, test_data):
    # Get the initial model from the server
    _group = [w for w in workers].append(0)
    group = dist.new_group(_group)

    for p in model.parameters():
        tmp_p = torch.zeros_like(p)
        dist.scatter(tensor=tmp_p, src=0, group=group)
        p.data = tmp_p
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
    for epoch in range(args.epochs):
        batch_interval = 0.0
        batch_comp_interval = 0.0
        batch_comm_interval = 0.0
        s_time = time.time()
        model.train()

        # Reduce the learning rate LR in some specific epochs
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
            # Synchronization
            # send epoch train loss firstly
            dist.gather(loss.data, dst = 0, group = group)
            for idx, param in enumerate(model.parameters()):
                dist.gather(tensor=delta_ws[idx], dst=0, group=group)
                recv = torch.zeros_like(delta_ws[idx])
                dist.scatter(tensor=recv, src=0, group=group)
                param.data = recv

            epoch_train_loss += loss.data.item()
            batch_end_time = time.time()

            batch_interval += batch_end_time - batch_start_time
            batch_comp_interval += batch_comp_time - batch_start_time
            batch_comm_interval += batch_end_time - batch_comp_time

            logs = torch.tensor([0.0, batch_interval/(batch_idx+1), batch_comp_interval/(batch_idx+1), batch_comm_interval/(batch_idx+1)])
            time_logs.write(str(logs) + '\n')
            time_logs.flush()

        print('Rank {}, Epoch {}, Loss:{}'
             .format(rank, epoch, loss.data.item()))

        e_time = time.time()
        #epoch_train_loss /= len(train_data)
        #epoch_train_loss = format(epoch_train_loss, '.4f')
        # test the model
        #test_loss, acc = test_model(rank, model, test_data, criterion=criterion)
        acc = 0.0
        batch_interval /= batch_idx
        batch_comp_interval /= batch_idx
        batch_comm_interval /= batch_idx
        logs = torch.tensor([acc, batch_interval, batch_comp_interval, batch_comm_interval])
        time_logs.write(str(logs) + '\n')
        time_logs.flush()
        #dist.gather(tensor=logs, dst = 0, group = group)
    time_logs.close()




def init_processes(rank, size, workers,
                   model, save_path,
                   train_dataset, test_dataset,
                   fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, workers, model, save_path, train_dataset, test_dataset)


if __name__ == '__main__':

    workers = [v+1 for v in range(args.workers_num)]

    if args.model == 'MnistCNN':
        model = MnistCNN()

        train_transform, test_transform = get_data_transform('mnist')

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                       transform=train_transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
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
    elif args.model == 'AlexNet':

        train_transform, test_transform = get_data_transform('cifar')

        if args.data_name == 'cifar10':
            model = AlexNetForCIFAR()
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        else:
            model = AlexNetForCIFAR(num_classes=100)
            train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False,
                                              transform=train_transform)
            test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False,
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
    test_bsz = 400

    train_bsz /= len(workers)
    train_bsz = int(train_bsz)

    train_data = partition_dataset(train_dataset, workers)
    test_data = partition_dataset(test_dataset, workers)

    this_rank = args.this_rank
    train_data = select_dataset(workers, this_rank, train_data, batch_size=train_bsz)
    test_data = select_dataset(workers, this_rank, test_data, batch_size=test_bsz)

    # Initialize the test dataset
    #test_data = DataLoader(test_dataset, batch_size=test_bsz, shuffle=True)

    world_size = len(workers) + 1

    save_path = str(args.save_path)
    save_path = save_path.rstrip('/')

    p = TorchProcess(target=init_processes, args=(this_rank, world_size, workers,
                                                  model, save_path,
                                                  train_data, test_data,
                                                  run))
    p.start()
    p.join()
