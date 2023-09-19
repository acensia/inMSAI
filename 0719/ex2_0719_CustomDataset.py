import glob
import os.path
import cv2
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.file_list = glob.glob(os.path.join(data_dir,"*", "*.jpg"))
        self.transform = transform
        # print(self.file_list)

        data_type = os.path.basename(data_dir)
        if data_type == "train_set":
            labels = pd.read_csv(data_dir[:data_dir.find(data_type)] + "train_labels.csv")
        else:
            labels = pd.read_csv(data_dir[:data_dir.find(data_type)] + "val_labels.csv")
        print(type(self.file_list))
        print(len(self.file_list))

    def __getitem__(self, idx):
        pass


def main():
    D = CustomDataset("meal_set/train_set")

if __name__ == "__main__":
    main()
