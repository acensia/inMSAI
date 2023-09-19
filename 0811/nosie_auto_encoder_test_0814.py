import torch
import torchvision
import matplotlib.pyplot as plt


from noise_auto_encoder_model import DenoisingAutoEncoder

# transforms
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,),)
])

# device setting

# batch_size
batch_size = 10

# model loader
model = DenosingAutoEncoder()
model.load_state_dict(torch.load("./denosingAutoEncoder_mdoel.pt",map_location='cpu'))
model.eval()

#dataset, dataloader
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for imgs, _ in test_loader:

    noise_factor = 0.2
    noisy_imgs = imgs + noise_factor * torch.randn(imgs.size())

    reconstructed_imgs = model(noisy_imgs)

    for j in range(batch_size):
        fig, axes = plt.subplot(1,3,figsize=(15,5))
        og_imgs = imgs[j].view(28, 28)

        #og img
        axes[0].imshow(og_imgs, cmap='gray')
        axes[0].set_title("Og Img")

        #noisy img

        #reconstructed img