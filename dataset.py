import numpy as np
import torch.utils.data as data
import torch
from torchvision import transforms, datasets
import torchvision
from torch.utils.tensorboard import SummaryWriter


def get_dataset(batch_size, path):
    TRANSFORM_IMG = transforms.Compose([
                            transforms.Resize(128),
                            transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
    ])

    train_data = torchvision.datasets.ImageFolder(root=path, transform=TRANSFORM_IMG)
    data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
    # Compute Number of batches
    num_batches = np.ceil(len(train_data)/batch_size)

    return data_loader, num_batches

