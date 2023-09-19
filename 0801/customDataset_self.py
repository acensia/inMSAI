import json
import os
import numpy as np
import cv2
from torch.utils.data import Dataset


class KeyPointDataset(Dataset):
    def __init__(self, root, mode="train", transform=None) -> None:
        self.root = root
        self.transform = transform
        self.img_files = sorted(os.listdir(os.path.join(self.root)))

        self.annotations_files = sorted(os.listdir(os.path.join(self.root, "annotations")))

    def __getitem__(self, index) -> Any:
        img_path = os.path.join(self.root, "images", self.img_files[index])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[index])
        img_og = cv2.imread(img_path)
        img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)

        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            bboxes_og = data["bboxes"]
            keypoints_og = data["keypoints"]

        if self.transform:

            keypoints_og_flattened = [el[0:2] for kp in keypoints_og for el in kp]

            transformed = self.transform(image=img_og, bboxes=bboxes_og,
                                         bboxes_labels=bboxes_labels_og,
                                         keypoints=keypoints_og_flattened)
            
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed["keypoints"]), (-1, 2, 2)).tolist()

            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened):
                obj_keypints = []

                



        return img, label

    
    def __len__(self):
        pass


if __name__ == "__main__":
    root_path = "./keypoint_dataset/"

    dataset = KeyPointDataset(f"{root_path}train")