import pandas as pd
import PIL
from sklearn.model_selection import KFold
from torchvision import transforms


class ImageTransform:
    def __init__(self, resize: int, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.data_transform: dict[str, transforms.Compose] = {
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

    def __call__(self, img: PIL.Image, phase: str = "train") -> None:
        return self.data_transform[phase](img)
