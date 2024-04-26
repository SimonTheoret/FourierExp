from typing import Optional
from torch.utils.data import DataLoader

from art.estimators.classification import PyTorchClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD

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
    batch_size: int = 2048,  # TODO: change this
    lr: float = 1e-3,
    n_epochs: int = 105,
    adv: bool = False,  # do we need to have an adv training ?
    seeds_range: int = 6,
    from_checkpoint: Optional[int] = None,
) -> None:
    orig_exp_name = exp_name.lower()
    model_name = model_name.lower()
    optim_name = optim_name.lower()
    dataset_name = dataset_name.lower()
    if adv:
        print("Doing adversarial training")
    for seed in range(seeds_range):
        torch.manual_seed(seed)
        exp_name = orig_exp_name + "_seed_" + str(seed)
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
                model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
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
        if from_checkpoint is not None:
            trainer.load_data(from_checkpoint)

        print(f"device is: {trainer.device}")

        if adv:
            classifier = PyTorchClassifier(
                model=trainer.model,
                clip_values=(0.0, 1.0),
                preprocessing=None,
                loss=loss_func,
                optimizer=trainer.optimizer,
                input_shape=(3, 32, 32),
                nb_classes=10,
            )

            epsilon = 8.0 / 255.0
            adv_trainer = AdversarialTrainerMadryPGD(classifier, eps=epsilon)

            # # Build a Keras image augmentation object and wrap it in ART
            # assert dataset.train_dataloader is not None
            # art_datagen = PyTorchDataGenerator(
            #     iterator=dataset.train_dataloader,
            #     size=len(dataset.train_dataloader) * batch_size,
            #     batch_size=batch_size,
            # )

            # Step 5: fit the trainer
            assert trainer.dataset.train_dataset is not None

            # hack to make have data as a numpy array with the transforms:
            new_train_dataloader = DataLoader(
                trainer.dataset.train_dataset,
                batch_size=len(trainer.dataset.train_dataset),
            )
            data = next(iter(new_train_dataloader))[0].numpy()
            labels = next(iter(new_train_dataloader))[1].numpy()
            # hack has ended

            while trainer.current_epoch < n_epochs:
                trainer.device = torch.device("cpu")
                adv_trainer.fit(x=data, y=labels, nb_epochs=1, device = "cpu")
                trainer.test()
                if trainer.current_epoch % 15 == 0 or trainer.current_epoch == 1:
                    with torch.no_grad():
                        trainer.device = torch.device("cpu")
                        trainer.compute_fourier_low_pass_accuracy()
                        trainer.compute_fourier_high_pass_accuracy()

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

            trainer.save_data(
                exp_name,
                model_name,
                optim_name,
                dataset_name,
                batch_size,
            )

        if not adv:
            while trainer.current_epoch < n_epochs:
                trainer.train()
                trainer.test()
                if trainer.current_epoch % 15 == 0 or trainer.current_epoch == 1:
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
