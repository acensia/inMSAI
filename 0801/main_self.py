import torch
import torchvision
import albumentations as A

from engine import train_one_epoch
from utils import collate_fn

from torch.utils.data import DataLoader
from Customdataset import KeypointDataset

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_model(num_keypoints, weights_path=None):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), 
                                       aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))

    model = keypointrcnn_resnet50_fpn(pretrained=False,
                                      pretrained_backbone=True,
                                      num_keypoints=num_keypoints,
                                      num_classes=2, #무조건 배경 class 포함
                                      rpn_anchor_generator=anchor_generator)
    
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model


train_transform = A.Compose([
    A.Sequential([
        A.RandomRotate90(p=1),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True,
                                   always_apply=False, p=1)
    ], p=1)
], keypoint_params=A.KeypointParams(format="xy"),
bbox_params=A.BboxParams(format="pascal_voc", label_fields=['bboxes_labels']))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

KEYPOINTS_FOLDER_TRAIN = "./keypoint_dataset/train/"

train_dataset = KeypointDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform)
train_dataloader= DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)

model = get_model(num_keypoints=2)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

num_epochs = 20


for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=1000)

    lr_scheduler.step()

    if epoch%10 == 0:
        torch.save(model.state_dict(), f"./keypointsrcnn_weights_{epoch}.pth")



torch.save(model.state_dict(), "./keypointsrcnn_weights_last.pth")