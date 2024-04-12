from fire import Fire

from exp_datasets import VanillaCifar10
from architecture import GenericTrainer
from models.allcnn import AllConvNet
from models.mobilevit import MobileViT
import torch
from torchinfo import summary

torch.manual_seed(42)


def main_generic() -> None:
    batch_size = 1024
    dataset = VanillaCifar10(batch_size=batch_size)
    dataset.build_dataset()  # download and setups the VanillaCifar10 object
    model = AllConvNet()
    # model = MobileViT()
    if isinstance(model, MobileViT):
        print(
            f"Number of trainable parameters {model.model.num_parameters(only_trainable = True)}"
        )
        print(
            f"Number of parameters {model.model.num_parameters(only_trainable = False)}"
        )
    else:
        summary(model, input_size=(batch_size, 3, 32, 32))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.15, momentum=0.9, weight_decay=0.001, 
    )
    loss_func = torch.nn.CrossEntropyLoss()
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
    print(f"device is: {trainer.device}")
    for i in range(100):
        trainer.train()
        trainer.test()
        # if i % 1000 == 0:
        #     with torch.no_grad():
        #         trainer.device = torch.device("cpu")
        #         trainer.compute_fourier_low_pass_accuracy()
        #         trainer.compute_fourier_high_pass_accuracy()
        #         trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # trainer.save_data()
    

if __name__ == "__main__":
    Fire(main_generic)
