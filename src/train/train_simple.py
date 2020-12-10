import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import mlflow
from src.models.simple import DenseNet
from nist_pytorch.data_loader import TandemDataset
from nist_ms.config import BaseMSConfig, TandemMLConfig


def main():
    config = TandemMLConfig()
    config.bin_size = 1.0
    config.max_mz = 500.0

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--data',
                        default='/home/lyg/source/deep/spectra/ei/nistms2/nist_ms/utils/human_hcd_tryp_good.pkl.gz',
                        help='pandas dataframe containing spectra and peptides')
    parser.add_argument('--output', default='test.pt', help='model output')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8123'
    mp.spawn(train, nprocs=args.gpus, args=(args, config))


def train(gpu, args, config):
    if gpu == 0:
        mlflow.log_params(vars(args))
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model = DenseNet(config)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    df = pd.read_pickle(args.data)
    df = df.query('set == "train"')

    train_dataset = TandemDataset(df.index, df, config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
                step = epoch * len(train_loader) + i
                mlflow.log_metric('train_loss', loss.data.item(), step)
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        torch.save(model.state_dict(), "test.pt")
        mlflow.log_artifact("test.pt")


if __name__ == '__main__':
    main()
