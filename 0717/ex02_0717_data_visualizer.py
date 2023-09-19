import os
import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = {}
        self.val_data = {}
        self.train_data = {}
        self.test_data = {}


    def load_data(self):
        train_dr = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'validation')
        test_dir = os.path.join(self.data_dir, 'test')

        for label in os.listdir(train_dr):
            label_dir = os.path.join(train_dr, label)
            # print(os.listdir(label_dir))
            cnt = len(os.listdir(label_dir))
            self.data[label] = cnt

        # print(self.data)

        for label in os.listdir(val_dir):
            label_dir = os.path.join(val_dir, label)
            # print(os.listdir(label_dir))
            cnt = len(os.listdir(label_dir))
            self.val_data[label] = cnt

        for label in os.listdir(val_dir):
            label_dir = os.path.join(test_dir, label)
            # print(os.listdir(label_dir))
            cnt = len(os.listdir(label_dir))
            self.test_data[label] = cnt
            if label in self.data:
                self.data[label] += cnt
            else :
                self.data[label] = cnt


    def visualize_data(self):
        labels = list(self.data.keys())
        cnts = list(self.data.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, cnts)
        plt.title("label data num")
        plt.xlabel("labels")
        plt.ylabel("data num")
        plt.xticks(rotation=45, ha='right', fontsize=8)

        plt.show()



if __name__:
    test = DataVisualizer("./food_dataset")
    test.load_data()
    test.visualize_data()