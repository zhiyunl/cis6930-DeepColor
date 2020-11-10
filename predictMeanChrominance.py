import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import nn

from FaceDataset import FaceDataset, AverageMeter, to_rgb
from models import *
from models.BaseNet import BaseNet


def train(train_loader, model, criterion, optimizer, epoch):
    print('Starting training epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    losses = AverageMeter()

    end = time.time()
    for i, (input_gray, input_ab) in enumerate(train_loader):
        # Use GPU if available
        input_gray, input_ab = input_gray.to(device), input_ab.to(device)

        # calculate mean of a b channel
        mean_ab = torch.mean(input_ab, dim=[2, 3])
        # print(mean_ab.shape)

        # Run forward pass
        output_ab = model(input_gray)
        # print(output_ab.shape)
        loss = criterion(output_ab, mean_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), loss=losses))

    print('Finished training epoch {}'.format(epoch))


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loadSize = 256
    fineSize = 128
    batch_size = 64

    # Training
    transform_trn = transforms.Compose([
        transforms.RandomChoice([
            transforms.Resize((loadSize, loadSize), interpolation=1),
            transforms.Resize((loadSize, loadSize), interpolation=2),
            transforms.Resize((loadSize, loadSize), interpolation=3)]),
        transforms.RandomChoice([transforms.RandomResizedCrop(fineSize, interpolation=1),
                                 transforms.RandomResizedCrop(fineSize, interpolation=2),
                                 transforms.RandomResizedCrop(fineSize, interpolation=3)]),
        transforms.RandomHorizontalFlip()]
    )

    dataset_trn = FaceDataset('./data/train', transform_trn)
    train_loader = torch.utils.data.DataLoader(dataset_trn, batch_size=batch_size, shuffle=True)

    # Validation
    val_transforms = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128)])
    dataset_val = FaceDataset('./data/valid', val_transforms)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False)

    dataset_size = len(dataset_trn)
    print('#training images = %d' % dataset_size)

    model = BaseNet().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Make folders and set parameters
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    save_images = True
    best_losses = 1e10
    epochs = 1000
    save_epoch = 50
    val_epoch = 30
    cnt_save = 0
    cnt_val = 0
    # Train model
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch)
