from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Self, Tuple, Any
from deprecation import deprecated
from cifar10 import Cifar10

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from art.attacks.attack import EvasionAttack

from utils.images import BatchedImages


class Dataset(ABC):
    """
    Abstract class for a dataset an its transformation.
    """

    @abstractmethod
    def download_raw_dataset(self):
        """
        Downloads the dataset and sets the dataset
        attribute.
        """
        pass

    @abstractmethod
    def save_dataset(self):
        """
        Saves the dataset into the data_root folder under the name
        `self.dataset_name`.
        """
        pass

    @abstractmethod
    def build_dataset(self) -> None:
        """
        Build the dataset by applying the transformations and the
        appropriate functions, if any. This function updates the
        raw_dataset, the train_dataset, the test_dataset, the
        train_dataloader and the test_dataloader attributes. It sets
        the transformation in the transforms attributes of the
        datasets. It saves the newly created dataset.
        """
        pass

    @abstractmethod
    def split_train_test(self, src: str) -> None:
        """
        Splits its current internal dataset into two Datasets. It
        updates the train_torch_dataset and test_torch_dataset
        attributes.
        """
        pass

    @deprecated
    @abstractmethod
    def apply_transformation(
        self, images: npt.ArrayLike | torch.Tensor
    ) -> npt.ArrayLike | torch.Tensor:
        """
        Apply transformation, such as flip and crop, to a batch of
        images.
        """
        pass

    @abstractmethod
    def apply_gaussian(
        self, images: npt.ArrayLike | torch.Tensor, to_tensor: bool = True
    ) -> npt.ArrayLike | torch.Tensor:
        """
        Apply gaussian transformation and returns the new images as
        tensors. The value is clipped to `[0,1]`.

        Parameters
        ----------
        images: npt.ArrayLike | torch.Tensor.
            Images on which the gaussian transformation will be applied.
        to_tensor: bool, defaults to True.
            Determines if the return value should be a torch.Tensor.
        """
        pass

    @deprecated
    @abstractmethod
    def generate_adversarial(
        self, images: npt.ArrayLike | torch.Tensor, to_tensor: bool = True
    ) -> npt.ArrayLike | torch.Tensor:
        """
        Generates the adversarial examples.

        Parameters
        ----------
        images: npt.ArrayLike | torch.Tensor.
            Images for which the adversarial examples will be computed.
        to_tensor: bool, defaults to True.
            Determines if the return value should be a torch.Tensor.
        """
        pass

    @abstractmethod
    def next_train_batch(self) -> BatchedImages:
        """
        Returns a batch of images from the training set, i.e. a
        BatchedImages object. These images are capable of computing
        their own Fourier transformation.
        """
        pass

    @abstractmethod
    def next_test_batch(self) -> BatchedImages:
        """
        Returns a batch of images from the test set, i.e. a
        BatchedImages object. These images are capable of computing
        their own Fourier transformation.
        """


# Type aliases
Float = float | torch.Tensor | np.ndarray


@dataclass
class Trainer(ABC):
    max_epochs: int
    train_dataset: Cifar10
    test_dataset: Cifar10
    current_epoch: int
    loss_func: Any
    optimizer: torch.optim.Optimizer
    model: nn.Module
    _batch: Optional[BatchedImages]
    attack: Optional[EvasionAttack]
    train_accuracy: Optional[list[Float]] = None
    test_accuracy: Optional[list[Float]] = None
    current_loss: Float = 0
    save_dir: str = "models/"
    log_interval: int = 4000
    device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def train(self) -> None:
        """
        Trains the model.
        """
        pass

    @abstractmethod
    def test(self) -> None:
        """
        Tests the model on the test dataset.
        """
        pass

    def save_model(self) -> None:
        """
        Saves the model localy.
        """

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
            self._batch = self.train_dataset.next_train_batch()
        if from_test:
            self._batch = self.test_dataset.next_test_batch()
