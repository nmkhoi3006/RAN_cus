from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (Compose, RandomErasing, GaussianBlur,
                                     ToTensor, Normalize, Resize, RandomAffine, 
                                     ColorJitter, RandomResizedCrop, RandomHorizontalFlip,
                                     RandomRotation)
import torch
import numpy as np
import os
import cv2

# from model import Resnet
# from config import get_config

class AIODataset(Dataset):
    def __init__(self, root, split, transform_image=None, transform_label=None):
        super().__init__()
        self.path_dir = os.path.join(root, split)

        self.class_name = {"NM": 0, "BN": 1, "DG" : 2, "LC": 3}
        self.image_path = []
        self.label = []
        self.transform_image = transform_image
        self.transform_label = transform_label
        for path in os.listdir(self.path_dir):
            for image in os.listdir(os.path.join(self.path_dir, path)):
                self.image_path.append(os.path.join(self.path_dir, path, image))
                self.label.append(self.class_name[path])

    def __len__(self):
        return len(self.image_path)
    
    def one_hot(self, label):
        matrix = np.zeros(len(self.class_name))
        matrix[label] = 1

        matrix = torch.tensor(matrix)
        return matrix

    def __getitem__(self, idx):
        image_path, label = self.image_path[idx], self.label[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label = self.one_hot(label=label)
        if self.transform_image:
            image = self.transform_image(image)

        if self.transform_label:
            label = self.transform_label(label)
        return image, label
    
class AIOTest(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_path = []
        for image in os.listdir(self.root):
            self.image_path.append(os.path.join(self.root, image))
    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image_id = os.path.basename(image_path)
        image_id = image_id.split('.')[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image_id, image
    
if __name__ == "__main__":
    # config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform = Compose([
    #     ToTensor(),
    #     Resize((224, 224)),
    #     # GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    #     RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    #     ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
    #     # RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10)
    # ])

    # transform = Compose([
    #     ToTensor(),
    #     RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    #     RandomHorizontalFlip(p=0.2),
    #     RandomRotation(degrees=45),
    #     ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
    #     GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    #     RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    # ])


    transform = Compose([
        ToTensor(),
        Resize((664, 664)),
    ])
    ds = AIODataset(root='./aio-hutech', split="train", transform_image=transform)
    image, label = ds[10]

    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    print(image.shape)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    print(label)

    # image = image.unsqueeze(0)

    # model =  Resnet(
    #     in_channels=3, 
    #     num_classes=4,
    #     num_blocks=config["num_blocks"],
    #     num_channels=config["num_channels"],
    #     num_heads=config["num_heads"],
    #     dropout=config["dropout"],
    #     device=device
    # ).to(device)


    # pre_train = torch.load(config["pre_train_path"], map_location=torch.device('cpu'))
    # model.load_state_dict(pre_train['model_state_dict'])

    # out = model(image)
    # print(out)
