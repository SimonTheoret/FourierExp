from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from utils.images import BatchedImages


class Dataset(ABC):
    """
    Abstract class for a dataset an its transformation.
    """

    @abstractmethod
    def download_dataset(self, destination: str):
        """
        Downloads the dataset in local memory and sets the dataset
        attribute.
        """
        pass

    @abstractmethod
    def load_dataset(self, src: str):
        """
        Loads the dataset from the source src.
        """
        pass

    @abstractmethod
    def split_train_test(self, src: str) -> Tuple[Self, Self]:
        """Splits its current internal dataset into two Datasets"""
        pass

    @abstractmethod
    def apply_transformation(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation, such as flip and crop, to a batch of
        images.
        """
        pass

    @abstractmethod
    def apply_gaussian(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian transformation and returns the new images as
        tensors.
        """
        pass

    @abstractmethod
    def generate_adversarial(self, images: torch.Tensor) -> torch.Tensor:
        """
        Generates the adversarial examples.
        """
        pass

    @abstractmethod
    def next_batch(self) -> BatchedImages:
        """
        Returns a batch of images, i.e. a BatchedImages
        object. These images are capable of computing their own
        Fourier transformation.
        """
        pass

# Type aliases
Float = float | torch.Tensor | np.ndarray


@dataclass
class Trainer(ABC):
    loss: Float
    max_epochs: int
    train_dataset: Dataset
    test_dataset: Dataset
    epoch: int
    train_accuracy: Optional[Float]
    test_accuracy: Optional[Float]
    optimizer: torch.optim.Optimizer
    model: nn.Module
    _batch: BatchedImages

    @abstractmethod
    def train(self, model: nn.Module):
        """
        Trains the model.
        """
        pass

    @abstractmethod
    def test(self, model: nn.Module):
        """
        Tests the model on the test dataset.
        """
        pass

    @abstractmethod
    def compute_metrics(self):
        """
        Computes the metrics.
        """

    def set_batch(self, new_batch: BatchedImages):
        """
        Sets the current batch _batch.
        """
        del self._batch
        self._batch = new_batch

    def get_batch(self):
        """
        Gets the current batch.
        """
        return self._batch

    def next_batch(self, from_train: bool, from_test: bool):
        """
        Updates the _batch attribute to the next batch. It can either
        be from the train dataset (self.train_dataset) or the test
        dataset (self.test_dataset). from_train and from_test cannot
        be equal.
        """
        assert from_train != from_test
        if from_train:
            self._batch = self.train_dataset.next_batch()
        if from_test:
            self._batch = self.test_dataset.next_batch()
