import albumentations as A
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from SegDataset import customVOCSegmentation


if __name__ == "__main__":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    train_transforms = A.Compose([
        A.Resize(520, 520),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])