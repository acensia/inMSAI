import cv2
import albumentations as A
from torch.utils.data import DataLoader
from utils import collate_fn
from Customdataset import KeypointDataset

train_transform = A.Compose([
    A.Sequential([
        A.RandomRotate90(p=1),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True,
                                   always_apply=False, p=1)
    ], p=1)
], keypoint_params=A.KeypointParams(format='xy'),
bbox_params=A.BboxParams(format="pascal_voc", label_fields=['bboxes_labels'])
)


root = "./keypoint_dataset/"
dataset = KeypointDataset(f"{root}train/",transform=train_transform, demo=True)

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
batch = next(iterator)

batch_ = None
for item in data_loader:
    batch_ = item
    break



def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])

        img = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

    for kpt in keypoints:
        for idx, kp in enumerate(kpt):
            img = cv2.circle(img, tuple(kp), 5, (255, 0, 0), 10)
            