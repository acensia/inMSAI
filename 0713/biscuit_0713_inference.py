import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models.mobilenetv2 import mobilenet_v2
from torch.optim import AdamW
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from biscuit_0713_customDataset import myCustomDataset
from biscuit_0713_train import train

from sklearn.model_selection import train_test_split

def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = mobilenet_v2(pretrained=True)
    in_features_ = 1280
    model.classifier[1] = nn.Linear(in_features_, 15)


    model.load_state_dict(torch.load("./PTs/ex01_0714_best_mobilenet_v2.pt"))
    model.to(device)
    # aug
    train_transforms = transforms.Compose([
        transforms.CenterCrop((244,244)),
        transforms.Resize((224,244)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=15),
        transforms.RandAugment(),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.CenterCrop((244, 244)),
        transforms.Resize((224, 244)),
        transforms.ToTensor()
    ])

    # dataset dataloader
    whole_data = myCustomDataset("./BiscuitWrap/Biscuit Wrappers Dataset")
    train_dataset, val_dataset = train_test_split(whole_data, test_size=0.2, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # print(len(whole_data))
    # print(len(train_dataset))
    # print(len(val_dataset))

    exit()
    # loss function optimizer, epochs
    epochs = 20
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(),lr=0.001 , weight_decay=1e-2)
    train(model,train_loader,val_loader,epochs,optimizer,criterion,device)

if __name__ == "__main__" :
    main()