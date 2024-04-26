from exp_datasets import GaussianCifar10, VanillaCifar10
import torch
from torchvision.transforms.functional import rgb_to_grayscale as gray
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    gaussian_images = GaussianCifar10(batch_size=1)
    vanilla_images = VanillaCifar10(batch_size=1)
    gaussian_images.build_dataset()
    vanilla_images.build_dataset()
    gaussian_images, _ = gaussian_images.test_images()
    vanilla_images, _ = vanilla_images.test_images()
    gaussian_images.fourier_transform_all()
    vanilla_images.fourier_transform_all()
    assert (
        gaussian_images.fourier_tensor is not None
        and vanilla_images.fourier_tensor is not None
    )
    gaussian_avg = gray(torch.mean(gaussian_images.fourier_tensor, dim=0)).squeeze()
    vanilla_avg = gray(torch.mean(vanilla_images.fourier_tensor, dim=0)).squeeze()
    # plt.imshow(gaussian_avg.real.cpu(), cmap='gray')
    sns.heatmap(gaussian_avg.real.cpu(), cmap="gray", vmin=0, vmax=1)
    plt.title("Average frequencies of test set with Gaussian augmentation")
    plt.savefig("assets/freq_fourier_transf_gaussian.png")
    plt.show()
    sns.heatmap(vanilla_avg.real.cpu(), cmap="gray", vmin=0, vmax=1)
    plt.title("Average frequencies of test set without Gaussian augmentation")
    plt.savefig("assets/freq_fourier_transf_vanilla.png")
    plt.show()
    # sns.heatmap(gaussian_avg_low.real.cpu(), cmap="gray", vmin=0, vmax=1)
    # plt.show()
    # sns.heatmap(vanilla_avg_low.real.cpu(), cmap="gray", vmin=0, vmax=1)
    # plt.show()
    # sns.heatmap(gaussian_avg_high.real.cpu(), cmap="gray", vmin=0, vmax=1)
    # plt.show()
    # sns.heatmap(vanilla_avg_high.real.cpu(), cmap="gray", vmin=0, vmax=1)
    # plt.show()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
