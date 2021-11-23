# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
tqdm.pandas()
import datetime

plt.ion()   # interactive mode

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(str("logs", datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()) + ".log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
            "test": transforms.Compose(
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
    def __init__(self, csv, transformer=None, phase="train"):
        self.data = csv
        self.transformer = transformer
        self.phase = phase
        self.labels = torch.eye(4)[self.data["Target"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open("data/images/" + self.data.loc[idx]["image_id"] + ".jpg")
        image = self.transformer(image, phase=self.phase)
        labels = torch.tensor(self.data.loc[idx]["Target"])
        # return {"images": image, "labels": labels}
        return image, labels


def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def imshow(inp, out, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.imsave(out, inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataloaders, class_names, device, out_dir, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                title = f'label: {class_names[labels[j].int()]}, predicted: {class_names[preds[j]]}'
                ax.set_title(title)
                filename = os.path.join(out_dir, (str(i) + "_" + str(j) + "_" + title + ".jpg"))
                imshow(inputs.cpu().data[j], filename)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def main():

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df = pd.read_csv("data/train.csv")
    # one-hot表現をラベル表現に変換する
    columns = ["healthy", "multiple_diseases", "rust", "scab"]
    target = []
    for i in tqdm(range(len(df))):
        target.append(columns[np.argmax(df.iloc[i].values[1:])])

    df["Target"] = target
    df["Target"] = LabelEncoder().fit_transform(df["Target"])

    # データの分割
    # kaggleのデータなので、train.csv内に正解データが存在しない
    # そのためtest.csvのデータをtrain,val,testに分割して利用する
    data_dir = 'data/images'
    # transformerは別途設定するので一旦None
    root_dataset = PlantDataset(df, transformer=ImageTransform(size, mean, std))
    root_size = len(root_dataset)
    test_size = int(0.1 * root_size)
    val_size = int(0.3 * (root_size - test_size))
    train_size = root_size - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(root_dataset, [train_size, val_size,test_size])
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)}
    logger.info(f"length of each datasets => train:[{len(train_dataset)}], val:[{len(val_dataset)}], test[{len(test_dataset)}]")

    # phaseの設定（trainのみデータ拡張を行う）
    train_dataset.phase = "train"
    val_dataset.phase = "val"
    test_dataset.phase = "test"

    class_names = list(map(lambda x: columns[x[1].int()], train_dataset))

    # Dataloaderの設定
    dataloaders = {
        "train":  torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        "val":  torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 4)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, device, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    visualize_model(model_ft, dataloaders, class_names, device, "out")
    model_ft_path = os.path.join("model", "model_ft_" + str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()) + ".pth")
    torch.save(model_ft.state_dict(), model_ft_path)

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 4)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, dataloaders, dataset_sizes, device, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    visualize_model(model_conv, dataloaders, class_names, device, "out")
    model_conv_path = os.path.join("model", "model_conv_" + str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()) + ".pth")
    torch.save(model_conv.state_dict(), model_conv_path)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
