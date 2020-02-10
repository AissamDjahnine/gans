# system
import os
import argparse
# torch
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.utils as vutils
from torchvision import transforms, datasets
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# import model, helper functions :
from models import *
from utils import *
from dataset import *
#visualization
from IPython import display
import matplotlib.pyplot as plt

import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main(params, sw):

    # define model, loss, optim :
    print("[+][+][+]Starting of training process[+][+][+]")
    os.makedirs(os.path.join(params.path+"_model/Checkpoints/generator"), exist_ok=True)
    os.makedirs(os.path.join(params.path+"_model/Checkpoints/discriminator"), exist_ok=True)
    # Create Network instances and init weights
    generator = GenerativeNet()
    generator.apply(init_weights)

    discriminator = DiscriminativeNet()
    discriminator.apply(init_weights)

    # Enable cuda if available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
    
    # Get Data:
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

    return generator, discriminator


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
    train_data, num_batches = get_dataset(args.batch_size, args.path)

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


