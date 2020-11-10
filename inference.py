import glob
import os

import torch
from torchvision.transforms import transforms

from FaceDataset import FaceDataset, to_rgb
from models.DeepColorNet import DeepColorNet

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    batch_size = 64
    # find the latest model file
    model_list = glob.glob('checkpoints/*.pth')
    model = DeepColorNet().to(device)
    model.load_state_dict(torch.load(sorted(model_list)[-1]))
    model.eval()
    # Validation
    val_transforms = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128)])
    dataset_val = FaceDataset('./data/valid', val_transforms)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    (input_gray, input_ab) = next(iter(val_loader))

    input_gray, input_ab = input_gray.to(device), input_ab.to(device)
    output_ab = model(input_gray)
    save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
    save_name = 'test.jpg'
    to_rgb(input_gray[0].cpu(), ab_input=output_ab[0].detach().cpu())
    # for i, (input_gray, input_ab) in enumerate(loader_val):
    #
