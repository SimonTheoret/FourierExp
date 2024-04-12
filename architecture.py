from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from art.attacks.attack import EvasionAttack
import numpy as np
import torch
import torch.nn as nn

from cifar10 import Cifar10
from utils.images import BatchedImages


# Type aliases
Float = float | torch.Tensor | np.ndarray


@dataclass
class GenericTrainer(ABC):
    exp_name: str
    max_epochs: int
    square_side_length: int
    dataset: Cifar10
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizer: torch.optim.Optimizer
    model: nn.Module
    # _batch: Optional[BatchedImages]
    attack: Optional[EvasionAttack]
    current_epoch: int = 0
    test_accuracy: Optional[list[Float]] = None
    all_losses: dict["str", list[Float]] = field(
        default_factory=lambda: {
            "train_loss": [],
            "test_loss": [],
            "adv_train_loss": [],
            "adv_test_loss": [],
            "fourier_high_pass_loss": [],
            "fourier_low_pass_loss": [],
        }
    )  # contains train loss, test loss and potentially adv train loss and adv test loss
    all_accuracies: dict["str", list[Float]] = field(
        default_factory=lambda: {
            "test_accuracy": [],
            "adv_test_accuracy": [],
            "fourier_high_pass_accuracy": [],
            "fourier_low_pass_accuracy": [],
        }
    )

    # train_loss: list[Float] = []
    # test_loss: list[Float] = []
    # train_accuracy: Optional[list[Float]] = None
    save_dir: str = "models/"
    log_interval: int = 5000
    device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
    ) -> None:
        """
        Trains the model. This function must be overriden for the ADVTrainer class.
        """
        # TODO: Compute accuracy at each epoch and add it to the train_accuracy list
        print("Training started")
        self.model.to(self.device)
        self.model.train()
        assert self.dataset.train_dataset is not None
        assert self.dataset.train_dataloader is not None
        print(f"device is: {self.device}")
        for batch_idx, (data, target) in enumerate(self.dataset.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.all_losses["train_loss"].append(loss.item())
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.current_epoch,
                        batch_idx * len(data),
                        len(self.dataset.train_dataset),
                        100.0 * batch_idx / len(self.dataset.train_dataset),
                        loss.item(),
                    )
                )

    def test(
        self,
    ) -> None:
        """
        Tests the model.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        self.model.to(self.device)
        assert self.dataset.test_dataset is not None
        assert self.dataset.test_dataloader is not None
        with torch.no_grad():
            for _, (data, target) in enumerate(self.dataset.test_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output: torch.Tensor = self.model(data)
                test_loss += self.loss_func(output, target)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataset.test_dataset)  # average test loss
        self.all_losses["test_loss"].append(test_loss)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(self.dataset.test_dataset),
                100.0 * correct / len(self.dataset.test_dataset),
            )
        )
        self.current_epoch += 1
        self.all_accuracies["test_accuracy"].append(
            100.0 * correct / len(self.dataset.test_dataset)
        )

    def save_model(self) -> None:
        """
        Saves the model locally.
        """
        torch.save(
            {
                "current_epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "losses": self.all_losses,
                "accuracies": self.all_accuracies,
            },
            self.save_dir + self.exp_name + f"_epoch{self.current_epoch}",
        )

    def batched_images(self) -> Tuple[BatchedImages, torch.Tensor]:
        """
        Returns the BatchedImages object from the current test set and
        the targets in the form of a torch.tensor. The fourier
        transform is calculated during the execution of this function.
        """
        batched, targets = self.dataset.test_images()
        batched.fourier_transform_all()
        return batched, torch.tensor(targets)

    def compute_fourier_low_pass_accuracy(self):
        batched, target = self.batched_images()
        batched.filter_low_pass(self.square_side_length)
        batched = batched.images_tensor
        assert batched is not None
        fourier_test_loss = 0
        correct = 0
        with torch.no_grad():
            self.model.eval()
            data, target = batched.to(self.device), target.to(self.device)
            output: torch.Tensor = self.model(data)
            fourier_test_loss += self.loss_func(output, target)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).item()

        fourier_test_loss /= int(data.shape[0])  # average test loss

        print(
            "\nFourier low pass test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                fourier_test_loss,
                correct,
                int(data.shape[0]),
                100.0 * correct / int(data.shape[0]),
            )
        )
        self.all_accuracies["fourier_low_pass_accuracy"].append(
            100.0 * correct / int(data.shape[0])
        )
        self.all_accuracies["fourier_low_pass_loss"].append(fourier_test_loss)

    def compute_fourier_high_pass_accuracy(self):
        batched, target = self.batched_images()
        batched.filter_high_pass(self.square_side_length)
        batched = batched.images_tensor
        assert batched is not None
        fourier_test_loss = 0
        correct = 0
        with torch.no_grad():
            self.model.eval()
            data, target = batched.to(self.device), target.to(self.device)
            output: torch.Tensor = self.model(data)
            fourier_test_loss += self.loss_func(output, target)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).item()

        fourier_test_loss /= int(data.shape[0])  # average test loss

        print(
            "\nFourier high pass test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                fourier_test_loss,
                correct,
                int(data.shape[0]),
                100.0 * correct / int(data.shape[0]),
            )
        )
        self.all_accuracies["fourier_high_pass_accuracy"].append(
            100.0 * correct / int(data.shape[0])
        )
        self.all_accuracies["fourier_high_pass_loss"].append(fourier_test_loss)
