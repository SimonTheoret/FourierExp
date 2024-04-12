from fire import Fire

from exp_datasets import VanillaCifar10
from architecture import GenericTrainer
from models.allcnn import AllConvNet
import torch


def main_generic() -> GenericTrainer:
    dataset = VanillaCifar10(batch_size=16)
    dataset.build_dataset()  # download and setups the VanillaCifar10 object
    model = AllConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss(reduction="sum")
    trainer = GenericTrainer(
        exp_name="testrun",
        max_epochs=1,
        square_side_length=2,
        dataset=dataset,
        optimizer=optimizer,
        model=model,
        loss_func=loss_func,
        attack=None,
    )
    for _ in range(3):
        trainer.train()
        trainer.test()

    # with torch.no_grad():
    #     trainer.device = torch.device("cpu")
    #     trainer.compute_fourier_low_pass_accuracy()
    #     trainer.compute_fourier_high_pass_accuracy()
    return trainer


if __name__ == "__main__":
    Fire(main_generic)
