import glob
import os
import cv2
from torch.utils.data import Dataset

class MyUSLicensePlateDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transform = transform
        self.label_dict = self.create_label_dict()

    def create_label_dict(self):
        label_dict = {}
        for file_path in label_dict:
            label = os.path.basename(os.path.dirname(file_path))
            if label not in label_dict:
                label_dict[label] = len(label_dict)

    def __getitem__(self, item):
        img_filepath = self.data_dir(item)
        img = cv2.imread(img_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = os.path.basename(os.path.dirname(img_filepath))
        label_idx = self.label_dict[label]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img, label_idx

# test = MyUSLicensePlateDataset("./US_license_plates_dataset/train", transform=None)
# for i in test:
#     pass