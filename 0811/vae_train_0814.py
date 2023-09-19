import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from vae_model_0814 import VAE
import torch.nn as nn

batch_size = 246
learning_rate = 0.0025
latent_dim = 20
num_epochs = 150

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
])


train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = VAE()

criterion = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.view(imgs.size(0), -1)
        optimizer.zero_grad()

        recon_imgs, mu, logval = model(imgs)

        recon_loss = criterion(recon_imgs, imgs) / batch_size

        kl_divergence = -0.5 * torch.sum(1 + logval - mu.pow(2) - logval.exp()) / batch_size

        loss = recon_loss + kl_divergence

        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss : {loss.item():.4f}")

torch.save(model.state_dict(), "./vae_model.pth")