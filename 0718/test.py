import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A

from torch.utils.data import DataLoader
from torchvision.models.resnet import  resnet50
from ex01_0714_CustomDataset import My_ex01_customdata

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50()
    model.fc = nn.Linear(2048, 20)
    model.load_state_dict(torch.load(f="./ex01_0717_resnet50_best.pt"))

    val_transforms = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])

    test_dataset = MyFoodDataset("./food_dataset/test/", transforms=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdims=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    print("test set : ACC {}/{} [{:.0f}]%\n".format((
        correct, len(test_loader.dataset),
        100*correct / len(test_loader.dataset)
    )))

if __name__ == "__main__":
    main()