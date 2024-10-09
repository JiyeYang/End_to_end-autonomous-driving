import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UdacityDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图片路径和相应的标签
        img_name = os.path.join(self.image_dir, str(self.data.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 获取对应的 angle, torque, speed
        labels = self.data.iloc[idx, 1:].values.astype('float32')
        return image, labels


# 数据加载函数
def get_loaders(csv_file, image_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = UdacityDataset(csv_file, image_dir, transform=transform)

    train_size = 50000
    val_size = len(dataset) - train_size

    #train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 手动划分，选择前 50000 张图片作为训练集
    train_set = torch.utils.data.Subset(dataset, list(range(train_size)))
    val_set = torch.utils.data.Subset(dataset, list(range(train_size, len(dataset))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
