import glob
import os.path
import cv2

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.file_list = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transform = transform
        # print(self.file_list)
        self.label_dict = {}
        folder_list = glob.glob(os.path.join(data_dir, "*"))

        for i, folder_name in enumerate(folder_list):
            label_name = os.path.basename(folder_name)
            self.label_dict[label_name] = i
    def __getitem__(self, idx):
        img = cv2.imread(self.file_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = os.path.dirname(self.file_list[idx])
        label = os.path.basename(label)

        return img, label


def main():
    D = CustomDataset("./오후-ex02_data/train")
    for i, l in D:
        print(l)

if __name__ == "__main__":
    main()

