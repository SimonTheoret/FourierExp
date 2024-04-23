import torch
from torchinfo import summary

from arch.allcnn import AllConvNet
from arch.mobilevit import MobileViT
from architecture import GenericTrainer
from exp_datasets import AdversarialCifar10, GaussianCifar10, VanillaCifar10


def main_generic(
    exp_name: str,
    model_name: str,
    optim_name: str,
    dataset_name: str,
    batch_size: int = 1024,
    lr: float = 1e-3,
    n_epochs: int = 300,
    adv: bool = False,  # do we need to have an adv training ?
    seeds_range: int = 8,
) -> None:
    exp_name = exp_name.lower()
    model_name = model_name.lower()
    optim_name = optim_name.lower()
    dataset_name = dataset_name.lower()
    for seed in range(seeds_range):
        torch.manual_seed(seed)

        # Dataset selection
        if dataset_name == "vanilla":
            dataset = VanillaCifar10(batch_size=batch_size)
        elif dataset_name == "gaussian":
            dataset = GaussianCifar10(batch_size=batch_size)
        else:
            dataset = AdversarialCifar10(batch_size=batch_size)
        dataset.build_dataset()  # download and setups the Dataset object

        # Model selection
        model = MobileViT() if model_name == "mobilevit" else AllConvNet()

        # Model summary
        if isinstance(model, MobileViT):
            print(
                f"Number of trainable parameters {model.model.num_parameters(only_trainable = True)}"
            )
            print(
                f"Number of parameters {model.model.num_parameters(only_trainable = False)}"
            )
        else:
            summary(model, input_size=(batch_size, 3, 32, 32))
        # Loss function
        loss_func = torch.nn.CrossEntropyLoss()
        # Optimizer selection
        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=lr,
            )
            if optim_name == "adamw"
            else torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
            )
        )
        # Trainer
        trainer = GenericTrainer(
            exp_name=exp_name,
            max_epochs=1,
            square_side_length=4,  # TODO: to include in the report
            dataset=dataset,
            optimizer=optimizer,
            model=model,
            loss_func=loss_func,
            attack=None,
        )
        print(f"device is: {trainer.device}")

        for i in range(n_epochs):
            trainer.train()
            trainer.test()
            if i % 15 == 0:
                with torch.no_grad():
                    trainer.device = torch.device("cpu")
                    trainer.compute_fourier_low_pass_accuracy()
                    trainer.compute_fourier_high_pass_accuracy()
                    trainer.device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                trainer.save_data(
                    exp_name,
                    model_name,
                    optim_name,
                    dataset_name,
                    batch_size,
                )

        with torch.no_grad():
            trainer.device = torch.device("cpu")
            trainer.compute_fourier_low_pass_accuracy()
            trainer.compute_fourier_high_pass_accuracy()
            trainer.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        trainer.save_data(
            exp_name,
            model_name,
            optim_name,
            dataset_name,
            batch_size,
        )


if __name__ == "__main__":
    from fire import Fire

    Fire(main_generic)
