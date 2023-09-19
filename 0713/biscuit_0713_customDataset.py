import glob
import os.path

from torch.utils.data import Dataset


class myCustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_list = []
        folder_list = glob.glob(os.path.join(data_dir, "*"))
        self.label_dict = {}
        self.label_list = []
        for i, folder in enumerate(folder_list):
            folder_name = os.path.basename(folder)
            self.label_dict[folder_name] = i
            labeled_data = glob.glob(os.path.join(folder, "*.jpg"))
            self.data_list += (labeled_data)
            self.label_list += [i for _ in range(len(labeled_data))]

        self.transform = None
        # print(self.label_list)

    def __getitem__(self, item):
        data_name = os.path.basename(self.data_list[item])
        return data_name, self.label_list[item]


