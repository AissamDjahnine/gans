# Import Librairies and packages :
import os
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from IPython import display
import torch
from torch import nn
from torch.optim import Adam
import  matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision import transforms, datasets
import time
import warnings
warnings.filterwarnings('ignore')

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

    return data_loader,num_batches


class DiscriminativeNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminativeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*4*4, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024*4*4)
        x = self.out(x)
        return x



class GenerativeNet(torch.nn.Module):

    def __init__(self):
        super(GenerativeNet, self).__init__()

        self.linear = nn.Linear(100, 1024*4*4)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Apply Tanh
        return self.out(x)
    

def real_data_target(size):
    '''
    Tensor containing ones, with shape = (size,3)
    '''
    data = Variable(torch.ones(size, 3))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = (size,3)
    '''
    data = Variable(torch.zeros(size, 3))
    if torch.cuda.is_available(): return data.cuda()
    return data

# weights initialisation
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

# create latent space vector
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available():
        return n.cuda()
    return n


def train_discriminator(discriminator,optimizer, real_data, fake_data,loss):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(discriminator, optimizer, fake_data,loss):
    # 2. Train Generator
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


def main(params,sw):

    # define model, loss, optim :
    print("[+][+][+]Starting of training process[+][+][+]")
    os.makedirs(os.path.join(params.path+"_model/Checkpoints/generator"), exist_ok=True)
    os.makedirs(os.path.join(params.path+"_model/Checkpoints/discriminator"), exist_ok=True)
    # Create Network instances and init weights
    generator = GenerativeNet()
    generator.apply(init_weights)

    discriminator = DiscriminativeNet()
    discriminator.apply(init_weights)

    #Enable cuda if available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
    
    #Get Data :
    train_data ,num_batches = get_dataset(params.batch_size, params.path)


    # Optimizers
    d_optimizer = Adam(discriminator.parameters(), lr=params.lr, betas=(0.5, 0.999))
    g_optimizer = Adam(generator.parameters(), lr=params.lr, betas=(0.5, 0.999))

    # Loss function
    loss = nn.BCELoss()
    
    # Number of test examples:
    num_test_samples = 8
    test_noise = noise(num_test_samples)



    starting_epoch = 0

    # Resume training from checkpoints :
    if os.path.exists(os.path.join(params.path+"_model/Checkpoints/discriminator", 'discriminator_training_state.pt')):
        checkpoint = torch.load(os.path.join(params.path+"_model/Checkpoints/discriminator", 'discriminator_training_state.pt'))
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        g_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        d_error = checkpoint['d_loss']

    if os.path.exists(os.path.join(params.path+"_model/Checkpoints/generator", 'generator_training_state.pt')):
        checkpoint = torch.load(os.path.join(params.path+"_model/Checkpoints/generator", 'generator_training_state.pt'))
        generator.load_state_dict(checkpoint['model_state_dict'])
        g_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        g_error = checkpoint['g_loss']




    for epoch in range(starting_epoch, params.epochs):
        print('==== EPOCH {}===='.format(epoch))
        for n_batch, (real_batch,_) in enumerate(train_data):

        # 1. Train Discriminator
            real_data = Variable(real_batch)
            if torch.cuda.is_available():
                real_data = real_data.cuda()

            # Generate fake data
            fake_data = generator(noise(real_data.size(0))).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator,d_optimizer, real_data, fake_data,loss)
            # 2. Train Generator
            # Generate fake data
            fake_data = generator(noise(real_batch.size(0)))
            # Train G
            g_error = train_generator(discriminator, g_optimizer, fake_data, loss)

            # Display Progress
            if (n_batch) % 15 == 0:
                print('Batch :[{}/{}]'.format(n_batch,num_batches))
                test_images = generator(test_noise).data
                grid = torchvision.utils.make_grid((test_images-test_images.min())/(test_images.max()-test_images.min()))
                sw.add_image('Generated Images', grid, epoch+1)
                sw.close()

        sw.add_scalar('Generator loss', g_error, epoch+1)
        sw.add_scalar('Discriminator loss', d_error, epoch+1)
        sw.add_scalar('G(x)', d_pred_real.mean(), epoch+1)
        sw.add_scalar('D(g(z))', d_pred_fake.mean(), epoch+1)
        print('Epoch: [{}/{}]'.format(epoch, params.epochs))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))
        print('Loss Generator : {:.4f}, Loss Discriminator :  {:.4f}'.format(g_error, d_error))

        if epoch % 25 == 0:
            #Save numpy array of test_images every 25 epochs :
            np.save('./Generated_Images_numpy_API/generated_image{}_{}'.format(epoch, params.epochs), test_images[0].cpu().numpy())

            # Checkpoint generator
            torch.save({
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'epoch': epoch,
                'g_loss': g_error,
            }, os.path.join(params.path+"_model/Checkpoints/generator", 'generator_training_state.pt'))

            # Checkpoint generator
            torch.save({
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': d_optimizer.state_dict(),
                'epoch': epoch,
                'd_loss': d_error,
            }, os.path.join(params.path+"_model/Checkpoints/discriminator", 'discriminator_training_state.pt'))

            print("[+][+][+] Checkpoint Epoch{} [+][+][+]".format(epoch))

    return generator,discriminator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default='', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=600, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.0002, type=float, metavar='LR', help='learning rate')

    args = parser.parse_args()

    sw = SummaryWriter('./runs')
    os.makedirs(os.path.join("Generated_Images_numpy_API"), exist_ok=True)
    train_data, num_batches = get_dataset(args.batch_size, args.path)

    # Save images grid to tensorboard :
    images, _ = next(iter(train_data))
    grid = torchvision.utils.make_grid((images-images.min())/(images.max()-images.min()))
    sw.add_image('Training Images', grid)
    sw.close()

    g, d = main(args, sw)
    torch.save(g.state_dict(), './generator_128_{}.pth'.format(args.epochs))
    print('Done training ( the model is saved as Generator_128_{}.pth)'.format(args.epochs))

    print('Testing the Model ( Generating 128 x 128 synthetic images ):')

    num_test_samples = 64
    test_noise = noise(num_test_samples)
    test_images = g(test_noise).data.cpu()
    # Grab a batch of real images from the dataloader
    train_data ,num_batches = get_dataset(args.batch_size, args.path)

    real_batch = next(iter(train_data))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=4, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Image")
    plt.imshow(np.transpose(vutils.make_grid(test_images[:], padding=4, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    plt.close()

    print("Done Testing")


