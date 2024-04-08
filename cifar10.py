from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple
from copy import copy

import numpy as np
import numpy.typing as npt
import torch
from torch import clip
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)

from architecture import Dataset

clipT = partial(clip, min=0.0, max=1.0)


@dataclass
class Cifar10(Dataset):
    """Abstract class for building a dataset. The only method to
    implement is the build_dataset method."""

    batch_size: int
    dataset_name: str
    transformations: Optional[list[Callable[[torch.Tensor], torch.Tensor]]]
    # train_dataset: Optional[CIFAR10 | TorchDataset] = None
    # test_dataset: Optional[CIFAR10 | TorchDataset] = None
    train_dataset: Optional[CIFAR10] = None
    test_dataset: Optional[CIFAR10] = None
    data_root: str = "data/"
    train_dataloader: Optional[DataLoader] = None
    test_dataloader: Optional[DataLoader] = None
    default_transformations: list[Callable[[torch.Tensor], torch.Tensor]] = [
        RandomCrop(size=32),  # size of the image (3x32x32)
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        clipT,
    ]

    seed: int = 42
    generator: Callable = torch.Generator().manual_seed
    gaussian_sigma: float = 0.1  # bound for the gaussian noise

    def build_dataset(self) -> None:
        self.download_raw_dataset()  # init the raw_dataset attribute
        self.split_train_test()  # init train_dataset and test_dataset attributes
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=False,
            )
        else:
            raise AttributeError("train_dataset is None")
        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=False,
            )
        else:
            raise AttributeError("test_dataset is None")

    def download_raw_dataset(self):
        if self.transformations is not None:
            transform = self.default_transformations + self.transformations
        else:
            transform = self.default_transformations

        self.train_dataset = CIFAR10(
            root=self.data_root,
            train=True,
            transform=Compose(transform),
            download=True,
        )

        self.test_dataset = CIFAR10(
            root=self.data_root,
            train=False,
            transform=Compose(transform),
            download=True,
        )

    def save_dataset(self):
        # if self.raw_train_dataset is not None:
        #     torch.save(self.raw_train_dataset, self.data_root + self.dataset_name)
        # else:
        raise NotImplementedError(
            "saving the dataset is automatic with download_dataset"
        )

    def split_train_test(self) -> None:
        raise NotImplementedError(
            "splitting the dataset is automatic with download_dataset"
        )
        # generator = self.generator(self.seed)
        # if self.raw_train_dataset is not None:
        #     split = random_split(
        #         self.raw_train_dataset, lengths=[0.8, 0.2], generator=generator
        #     )
        # else:
        #     raise AttributeError("No torch_dataset attribute")
        # self.train_dataset = split[0]
        # self.test_dataset = split[1]

    def apply_gaussian(
        self, images: npt.ArrayLike | torch.Tensor, to_tensor: bool = True
    ) -> npt.ArrayLike | torch.Tensor:
        """
        Applies the gaussian transformation on a tensor or a arraylike
        object. It returns a gaussian augmented image in the form of a
        tensor or the original object. It always clones the original
        images.

        Parameters
        ----------
        images: torch.Tensor | npt.ArrayLike
            The image to augment.
        to_tensor: bool
            If `True`, this function returns a tensor. Else, it
            returns an image of the same type as the `images`.
        """
        if not isinstance(images, torch.Tensor):
            imgs = torch.tensor(images) # Create a tensor
        else:
            imgs = torch.clone(images) # Copy the tensor
        generator = self.generator(self.seed)  # select generator with seed
        std_val = np.random.uniform(
            low=0, high=self.gaussian_sigma
        )  # sample the std_value, a float, given the bound `gaussian_sigma`
        std = torch.full_like(imgs, fill_value=std_val)  # Build the std matrix
        augmented_images = (
            torch.normal(mean=0.0, std=std, generator=generator) + imgs
        )  # add noise to images copy.
        augmented_images = torch.clip(augmented_images, max=1.0, min=0.0)
        if to_tensor:
            return augmented_images  # return image tensor
        else:
            return augmented_images.numpy()  # return numpy array

    def apply_gaussian_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies the gaussian transformation on a tensor image. It
        returns a gaussian augmented image in the form of a tensor.

        Parameters
        ----------
        images: torch.Tensor
            The image to augment
        """
        assert isinstance(images, torch.Tensor)
        augmented_image = self.apply_gaussian(images, to_tensor=True)
        if isinstance(augmented_image, torch.Tensor):
            return augmented_image
        else:
            raise ValueError(
                "apply_gaussian returned a npt.arraylike altough to_tensor = true"
            )

    def test_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the test data as a numpy array. It returns the data and the
        targets in a tuple.
        """
        assert self.test_dataloader is not None
        test_dl = copy(self.test_dataloader)
        data = []
        targets = []
        for i, (data_point, target) in enumerate(test_dl):
            data.append(data_point)
            targets.append(target)
        return np.array(data), np.array(targets)

    def train_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the train data as a numpy array. It returns the data and the
        targets in a tuple.
        """
        assert self.train_dataloader is not None
        train_dl = copy(self.train_dataloader)
        data = []
        targets = []
        for i, (data_point, target) in enumerate(train_dl):
            data.append(data_point)
            targets.append(target)
        return np.array(data), np.array(targets)
