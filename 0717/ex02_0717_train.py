import torch
import torch.nn as nn
import albumentations as A

from torchvision.models.mobilenetv2 import mobilenet_v2

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)


    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(1280, 20)
    model.to(device)

    train_transforms = A.Compose([

    ])