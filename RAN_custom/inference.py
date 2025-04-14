import os
import torch
from torch import nn
from dataset import AIOTest
from config import get_config
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import numpy as np
import random
from model import Resnet
from yolo.model import YoloClassifier

from pprint import pprint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  # NumPy seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)  # Nếu có nhiều GPU
    torch.backends.cudnn.deterministic = True  # Đảm bảo tái lập kết quả
    torch.backends.cudnn.benchmark = False  # Có thể làm chậm nhưng ổn định hơn



if __name__ == "__main__":
    set_seed(52)

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    order = config["order"]

    transform = Compose([ToTensor()])

    model = YoloClassifier(version=config["version"], c1=3, nc=config["num_classes"]).to(device)


    pre_train = torch.load(config["pre_train_best"], map_location=torch.device('cpu'))
    model.load_state_dict(pre_train['model_state_dict'])

    model.eval()
    ds = AIOTest(root='./aio-hutech/test', transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    softmax = nn.Softmax(dim=-1)
    predictions = {}
    for id, image in loader:
        image = image.to(device)
        output = model(image)
        output = softmax(output)


        predict_class = torch.argmax(output, dim=1)
        predictions[id[0]] = predict_class.item()
    # pprint(predictions)
    #Write prediction to file submission.csv
    data = {"id": [key for key in predictions.keys()], 
            "type": [value for value in predictions.values()],
    }
    
    df = pd.DataFrame(data)
    df.to_csv(f"predictions_{order}.csv", index=False)
    print("Done")