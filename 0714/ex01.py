import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from ex01_0714_CustomDataset import My_ex01_customdata

label_dict = {
    0:"Carpetweeds",
    1:"Crabgrass",
    2: "Eclipta",
    3: "Goosegrass",
    4: "Morningglory",
    5: "Nutsedge",
    6: "PalmerAmaranth",
    7: "PriclySida",
    8: "Purslane",
    9: "Ragweed",
    10: "Sicklepod",
    11: "SpottedSpurge",
    12: "SpurredAnoda",
    13: "Swinecress",
    14: "Waterhemp"
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)

    model = mobilenet_v2()
    in_features =1280
    model.classifier[1] = nn.Linear(in_features, 15)

    model.load_state_dict(torch.load(f="./model_pt/ex01_0714_best_mobilenet_v2.pt"))
    # model.load_state_dict(torch.load(f="./model_pt/ex02_0714_best_mobilenet_v2.pt"))

    val_transforms = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    val_dataset = My_ex01_customdata("./dataset/val", transforms=val_transforms)
    val_loader= DataLoader(val_dataset, batch_size=64, shuffle=False)

    model.to(device)
    model.eval()
    correct = 0
    #from tqdm import tqdm
    import cv2

    with torch.no_grad():
        for data, target, path in val_loader:
            target_ = target.item()

            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdims=True)

            target_label = label_dict[target_]
            pred_label = label_dict[pred.item()]
            true_label_txt = f"true : {target_label}"
            pred_label_txt = f"pred : {pred_label}"

            img = cv2.imread(path[0])
            img = cv2.resize(img, (500, 500))
            img = cv2.rectangle(img, (0,0), (500, 100), (255,2555,255),-1)
            # img = cv2.putText(img, pred_label_txt)

            cv2.imshow(img)
            exit()



if __name__ == "__main__":
    main()