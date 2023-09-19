import os

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from base_dcgan_model import BaseDCGANGenerator, BaseDCGANDiscriminator

dataroot = "./cat_dataset"
from base_dcgan_config import *


dataset = dset.ImageFolder(root = dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers= workers)

print("Dev is ", device)

if __name__ == "__main__":

    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(
    #     vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
    #     (1,2,0))
    # )
    #
    # plt.show()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, val=0)

    netG = BaseDCGANGenerator().to(device)
    netG.apply(weights_init)

    print(netG)

    netD = BaseDCGANDiscriminator().to(device)
    netD.apply(weights_init)

    print(netD)


    criterion = nn.BCELoss()

    fixed_noise = torch.randn(128, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.AdamW(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0


    output_images_path = "./generated_imgs"
    os.makedirs(output_images_path, exist_ok=True)

    print("Starting Training Loop....")

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label,
                               dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)

            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)

            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()


            if i % 50 == 0:
                print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) -1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


            if epoch % 5 == 0:
                weight_path = "./DCGAN_model_weight"
                os.makedirs(weight_path , exist_ok=True)
                torch.save(netG.state_dict(), f"{weight_path}/netG_epoch_{epoch}.pth")
                torch.save(netD.state_dict(), f"{weight_path}/netD_epoch_{epoch}.pth")

            if (epoch+1) % 5 == 0:
                vutils.save_image(img_list[-1], f"{output_images_path}/fake_images_epoch_{epoch}.png")
            iters += 1



