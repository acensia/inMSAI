import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models.mobilenetv2 import mobilenet_v2
from torch.optim import AdamW
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

def train(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    print("Train ....")
    for epoch in range(epochs) :
        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_acc = 0.0

        model.train()
        # tqdm
        train_loader_iter = tqdm(train_loader, desc=(f"Epoch : {epoch + 1}/{epochs}"), leave=False)

        for i, (data, target) in enumerate(train_loader_iter) :
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # acc
            _, pred = torch.max(outputs, 1)
            train_acc += (pred == target).sum().item()

            train_loader_iter.set_postfix({"Loss" :  loss.item()})

        train_loss /= len(train_loader)
        train_acc = train_acc / len(train_loader.dataset)

        # eval
        model.eval()
        with torch.no_grad() :
            for data, target in val_loader :
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
                val_loss += criterion(output, target).item()

        val_loss /= len(val_loader)
        val_acc = val_acc / len(val_loader.dataset)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # save model
        if val_acc > best_val_acc :
            torch.save(model.state_dict(), "./ex01_0714.pt")
            best_val_acc = val_acc
        print(f"Epoch [{epoch + 1} / {epochs}] , Train loss [{train_loss:.4f}],"
              f"Val loss [{val_loss :.4f}], Train ACC [{train_acc:.4f}],"
              f"Val ACC [{val_acc:.4f}]")

    torch.save(model.state_dict(), "./ex01_0174_last.pt")
    return  model, train_losses, val_losses, train_accs, val_accs