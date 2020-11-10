import numpy as np
import torch
from skimage.color import rgb2lab, rgb2gray, lab2rgb
from torchvision import datasets
import matplotlib.pyplot as plt


class FaceDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]

        img = self.loader(path)
        img_original = self.transform(img)
        img_original = np.asarray(img_original)
        # print(img_original.shape)
        img_lab = rgb2lab(img_original)
        # img_lab = (img_lab + 128) / 255
        img_ab = (img_lab[:, :, 1:3] + 128) / 255
        # HxWxab -> abxHxW
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
        # img_original = rgb2gray(img_original)
        img_l = img_lab[:, :, 0] / 100
        # img_l = (img_lab[:,:,0] +128) /255
        img_l = torch.from_numpy(img_l).unsqueeze(0).float()
        return img_l, img_ab


class AverageMeter(object):
    '''An easy way to compute and store both average and current values'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf()  # clear matplotlib
    img_abl = torch.cat((grayscale_input, ab_input), 0).numpy()  # combine channels
    img_lab = img_abl.transpose((1, 2, 0))  # rescale for matplotlib
    img_lab[:, :, 0:1] = img_lab[:, :, 0:1] * 100  # L channel range from 0-100
    img_lab[:, :, 1:3] = img_lab[:, :, 1:3] * 255 - 128  # ab channel range from -128 - 127
    color_image = lab2rgb(img_lab.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
    else:
        plt.imsave(arr=grayscale_input, fname='gray.jpg', cmap='gray')
        plt.imsave(arr=color_image, fname='color.jpg')
