import os
import time

import torch
import torchvision.transforms as transforms
from torch import nn

from FaceDataset import FaceDataset, AverageMeter, to_rgb
from models.DeepColorNet import DeepColorNet


def validate(val_loader, model, criterion, save_images, epoch):
    model.eval()

    # # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (input_gray, input_ab) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # Use GPU
        input_gray, input_ab = input_gray.to(device), input_ab.to(device)

        # Run model and record loss
        output_ab = model(input_gray)  # throw away class predictions
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Save a sample images to file
        if save_images:
            save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
            save_name = 'epoch-{}-img.jpg'.format(epoch)
            to_rgb(input_gray[0].cpu(), ab_input=output_ab[0].detach().cpu(), save_path=save_path,
                   save_name=save_name)

        # # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model loss
        if i % 2 == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.8f} ({loss.avg:.8f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

    # print('Finished validation.')
    return losses.avg


def train(train_loader, model, criterion, optimizer, epoch):
    # print('Starting training epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (input_gray, input_ab) in enumerate(train_loader):

        # Use GPU
        input_gray, input_ab = input_gray.to(device), input_ab.to(device)

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray)
        # print(input_ab.shape,output_ab.shape)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 2 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.8f} ({loss.avg:.8f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    # print('Finished training epoch {}'.format(epoch))


if __name__ == '__main__':
    # Init folders and set parameters
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loadSize = 256
    fineSize = 128
    # long time train
    # batch_size = 64
    # save_images = True
    # epochs = 1000
    # save_epoch = 100
    # val_epoch = 10
    # short time test
    batch_size = 16
    save_images = True
    epochs = 50
    save_epoch = 3
    val_epoch = 3

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
    loader_trn = torch.utils.data.DataLoader(dataset_trn, batch_size=batch_size, shuffle=True)

    # Validation
    transform_val = transforms.Compose([transforms.Resize(140), transforms.CenterCrop(128)])
    dataset_val = FaceDataset('./data/valid', transform_val)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False)

    # before augmentation
    print('#training images = %d' % len(dataset_trn))

    # setup CNN model
    model = DeepColorNet().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Train model
    cnt_save = 0
    cnt_val = 0
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(loader_trn, model, criterion, optimizer, epoch)
        # Save checkpoint and replace old best model if current model is better
        cnt_save += 1
        cnt_val += 1
        if cnt_val >= val_epoch:
            cnt_val = 0
            with torch.no_grad():
                losses = validate(loader_val, model, criterion, save_images, epoch)
        if cnt_save >= save_epoch:
            cnt_save = 0
            torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.8f}.pth'.format(epoch + 1, losses))
