import csv
import os
import time
import torch
import wandb
from nvidia import cudnn
from timm.data import RandAugment
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from tqdm import tqdm

from model.vit import VisionTransformer

best_acc = 0  # best test accuracy

def train():
    num_workers = 8
    batch_size = 30
    cifar_root_dir = '/home/user/datasets/cifar10'
    use_amp = False
    aug = False
    resume = False
    df=False
    dp = False
    n_epochs = 100
    opt = 'adam'
    lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    usewandb = True
    # Data
    print('==> Preparing data..')
    size = 384
    patch_size = 16
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Add RandAugment with N, M(hyperparameter)
    if aug:
        N = 2
        M = 14
        transform_train.transforms.insert(0, RandAugment(N, M))
    # Prepare dataset
    trainset = CIFAR10(cifar_root_dir,  download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=cifar_root_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Model factory..
    print('==> Building model..')
    net = VisionTransformer(img_size=size)
    if 'cuda' in device:
        print(device)
        if dp:
            print('using data parallel')
            net = torch.nn.DataParallel(net) # make parallel
            cudnn.benchmark = True
    if resume:
    # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    criterion = nn.CrossEntropyLoss()
    if opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    ##### Training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        progress = tqdm(trainloader)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress.update()
            progress.set_description(f'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return train_loss/(batch_idx+1)

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        progress = tqdm(testloader)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress.update()
                progress.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/{epoch}-epoch.pt')
            best_acc = acc
        os.makedirs('log', exist_ok=True)
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
        print(content)
        with open(f'log/log_patch{patch_size}.txt', 'a') as appender:
            appender.write(content + '\n')
        return test_loss, acc
    list_loss = []
    list_acc = []
    if usewandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project='cifar10-classification',
            # track hyperparameters and run metadata
            config={
                'learning_rate': lr,
                'architecture': 'transformer',
                'dataset': 'CIFAR-10',
                'epochs': n_epochs,
            }
        )
        wandb.watch(net, log='all', log_freq=500)
    net.cuda()
    for epoch in range(start_epoch, n_epochs):
        start = time.time()
        trainloss = train(epoch)
        val_loss, acc = test(epoch)
        scheduler.step(epoch-1) # step cosine scheduling
        list_loss.append(val_loss)
        list_acc.append(acc)
        # Log training..
        if usewandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': trainloss,
                'val_loss': val_loss,
                'val_acc': acc,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_time': time.time()-start})
        # Write out csv..
        with open(f'log/log_{net}_patch{patch_size}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)
    # writeout wandb
    if usewandb:
        wandb.save('wandb_{}.h5'.format(net))
if __name__ == '__main__':
    train()