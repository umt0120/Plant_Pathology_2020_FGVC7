import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.models import densenet121,inception_v3,densenet201,resnet152,resnet18
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
tqdm.pandas()


class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    # 訓練時のみデータオーグメンテーション
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase = "train") -> None:
        return self.data_transform[phase](img)

class PlantDataset(Dataset):
    def __init__(self, csv, transformer):
        self.data = csv
        self.transformer = transformer
        self.labels = torch.eye(4)[self.data["Target"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open("../data/images/" + self.data.loc[idx]["image_id"] + ".jpg")
        image = self.transformer(image)
        labels = torch.tensor(self.data.loc[idx]["Target"])
        return {"images": image, "labels": labels}


def fit(epochs, train_loader, valid_loader,  model, criteria, optimizer):
    for epoch in range(epochs + 1):
        training_loss = 0.0
        validation_loss = 0.0
        correct = 0.0
        total = 0.0

        print(f"{epoch + 1}/{epochs} Epochs")

        # training
        model.train()
        for batch_idx, d in enumerate(train_loader):
            data = d["images"]
            target = d["labels"]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criteria(output, target.long())
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]

            training_loss = training_loss + (1 + (batch_idx + 1) * (loss.data - training_loss))

            if batch_idx % 20 == 0:
                print(f"batch_Idx = [{batch_idx}], training_loss = [{training_loss}]")

            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total += data.size(0)
            print(f"batch_Idx = [{batch_idx}], training_accuracy = [{correct * 100 / total}]")

        # validation
        for batch_idx, d in enumerate(valid_loader):
            data = d["images"]
            target = d["labels"]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)
            loss = criteria(output, target.long())

            validation_loss = validation_loss + (1 / (batch_idx + 1)) * (loss.data - validation_loss)
            pred = output.data.max(1, keepdim=True)[1]
            if batch_idx % 20 == 0:
                print(f"batch_Idx = [{batch_idx}], validation_loss = [{validation_loss}]")

            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            total += data.size(0)
            print(f"batch_Idx = [{batch_idx}], validation_accuracy = [{correct * 100 / total}]")

def main():
    df = pd.read_csv("../data/train.csv")
    # one-hot表現をラベル表現に変換する
    columns = ["healthy", "multiple_diseases", "rust", "scab"]
    target = []
    for i in tqdm(range(len(df))):
        target.append(columns[np.argmax(df.iloc[i].values[1:])])

    df["Target"] = target
    df["Target"] = LabelEncoder().fit_transform(df["Target"])
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # transform = ImageTransform(size, mean, std)
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.496,0.456,0.406],[0.229,0.224,0.225])])
    train_dataset = PlantDataset(df, transform)
    # trainとtestを分割
    indices = range(len(train_dataset))
    split = int(0.1 * len(train_dataset))
    train_indices = indices[split:]
    test_indices = indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(train_dataset, sampler = train_sampler, batch_size=32)
    valid_loader = DataLoader(train_dataset,sampler = valid_sampler, batch_size=32)
    model = resnet18(pretrained=True)

    # 基底モデルは凍結
    for param in model.parameters():
        param.trainable = False

    # Linearレイヤの出力を4に変更
    model.fc = nn.Linear(512,4)
    fc_parameters = model.fc.parameters()

    # 付け替えたLinearレイヤは学習可能にする
    for param in fc_parameters:
        param.trainable = True

    if torch.cuda.is_available():
        model = model.cuda()

    # 損失関数
    criteria = nn.CrossEntropyLoss()
    # オプティマイザ
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    fit(20, train_loader, valid_loader, model, criteria, optimizer)


if __name__ == "__main__":
    main()
