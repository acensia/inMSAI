import argparse
import os
import torch


class Classifier_US_LicensePlate:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    def train(self, train_loader, val_loader, epochs, optimizer, criterion, start_epoch=0):
        best_val_acc = 0.0
        print("Training.....")

        for epoch in range(start_epoch, epochs):
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0
            for data, label in train_loader:


                print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss:.4f}, "

                      f"Val loss: {val_loss:.4f}, Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}")


def main() :
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./US_license_plates_dataset/train/",
                        help='directory path to the training dataset')
    parser.add_argument("--val_dir", type=str, default="./US_license_platest_dataset/val/",
                        help="directory path to the valid dataset")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=124,
                        help="batch size of training and validation")
    parser.add_argument("--learning rate", type=float, default=0.001,
                        help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="weight decay for optimizer")
    parser.add_argument("--resume_training", action="store_true",
                        help="resume training from the last checkpoint")
    parser.add_argument("--checkpoint-path", type=str,
                        default="./weight/0718/efficientnet_b0_checkpoint.pt",
                        help="path to the checkpoint file")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default="./weight/0718/")
    args = parser.parse_args()

    weight_folder_path = args.checkpoint_folder_path
    os.makedirs(weight_folder_path, exist_ok=True)