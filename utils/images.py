import torch
from dataclasses import dataclass
from typing import Tuple, Optional
from abc import ABC

@dataclass
class Image():
    dim: Tuple[int, int, int] | Tuple[int, int, int, int]
    original_images: torch.Tensor
    fourier_transform: Optional[torch.Tensor]

    def fourier_transform_single_image(self, img: torch.Tensor) -> torch.Tensor:
        """Utility function for applying the fourier transform on a
        single image with C channels. It applies the 2D FFT algorithm
        and then shifts the (recenter) the transform.

        Parameters
        ----------
        img: torch.Tensor
            Tensor of dimension (CxHxW). This must be a single image, not a batched image
        Returns
        -------
        Returns a torch.Tensor containing the fourier transform.
        """
        assert len(img.shape) == 3 #Asserts it contains a channel
        return torch.tensor([torch.fft.fftshift(torch.fft.fft2(img[i])) for i in range(img.shape[0])])

    def set_original_images(self, imgs: torch.Tensor) -> None:
        assert len(imgs.shape) == 4 # make sure images are batched
        


@dataclass
class BatchedImages(Image):
    dim: Tuple[int, int, int, int]

    def fourier_transform(self, imgs: torch.Tensor) -> None:
        """
        Update the content of the BatchedImages object with a tensor
        containing the results of the fourier transform on each of the
        images of the batch. The generated images are kept in the
        fourier_transform attribute of the object

        Parameters
        ----------
        img_batch: torch.Tensor
            Tensor of dimensions (NxCxHxW)
        """
        assert len(imgs.shape) == 4 # Asserts the images are batched
        batch_size = imgs.shape[0]
        self.original_images = torch.Tensor([self.fourier_transform_single_image(imgs[i]) for i in range(batch_size)])

