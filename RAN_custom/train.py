import torch
import os
import tqdm
import numpy as np
from torch.optim import Adam
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ColorJitter, RandomAffine
from sklearn.metrics import accuracy_score

from dataset import AIODataset
from model import ResidualAttentionModel 
from config import get_config


def trainer(config,
            loss_fn,
            model,
            optimizer,
            train_loader,
            val_loader,
            num_epochs,
            device):
    order = config["order"]
    save_folder = os.path.join(config["save_model"], f"save_{order}")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    best_acc = 0
    for epoch in range(num_epochs):
        progress_bar = tqdm.tqdm(train_loader)

        model.train()
        train_loss = []
        for image, label in progress_bar:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(image)
    
            loss = loss_fn(output, label)
            # loss.backward()
            # optimizer.step()

            train_loss.append(loss.item())
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.mean(train_loss):0.3f}")


        # VALIDATION
        model.eval()
        val_loss = []
        labels = []
        predictions = []
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device)
                label = label.to(device)

                output = model(image)
                loss = loss_fn(output, label)

                val_loss.append(loss.item())

                predict_class = torch.argmax(output, dim=1)
                predictions.extend(predict_class.cpu().numpy())
                labels.extend(label.cpu().numpy())

        acc = accuracy_score(labels, predictions)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {np.mean(val_loss):0.3f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {acc:0.3f}")
        
        check_point ={
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': acc,
        }

        torch.save(check_point, os.path.join(save_folder, "last_model.pth"))

        if acc > best_acc:
            best_acc = acc
            torch.save(check_point, os.path.join(save_folder, 'best_model.pth'))
    


if __name__ == "__main__":
    config = get_config()

    transform_train = Compose([
        ToTensor(),
        ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
        RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(1.2, 1.5), shear=10)
    ])

    transform_val = ToTensor()
    ds = AIODataset(root=config["root"], split='train')

    len_train = int(0.9 * len(ds))
    len_val = len(ds) - len_train
    train_dataset, val_dataset = random_split(ds, [len_train, len_val])

    train_dataset.dataset.transform_image = transform_train
    val_dataset.dataset.transform_image = transform_val


    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], num_workers=config["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], num_workers=config["num_workers"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualAttentionModel().to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = Adam(model.parameters(), lr=config["lr"])
    trainer(config,
            loss_fn,
            model,
            optimizer,
            train_loader,
            val_loader,
            config["num_epochs"],
            device=device)
    print(model)

