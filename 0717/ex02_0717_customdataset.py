import os.path


class MyCustomDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir



    def create_label(self):
        label_dict = {}
        for filepath in self.data_dir:
            label = os.path.basename(os.path.dirname(filepath))
