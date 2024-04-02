from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Iterable
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch


@dataclass(init = False)
class Image():
    """Class for a single image. Contains the dimension of the image,
    the original image and optionaly the fourier transform of the
    original image."""
    dim: Tuple[int, int, int]
    original_image: torch.Tensor
    fourier_transform: Optional[torch.Tensor]

    def __init__(self, image: npt.ArrayLike, dim: Tuple[int, int, int]) -> None:
        """Builds a Image object"""
        self.original_image = torch.tensor(np.array(image))
        self.dim = dim
        assert self.original_image.shape == self.dim

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
        assert len(img.shape) == 3 # Asserts it contains a channel
        return torch.tensor([torch.fft.fftshift(torch.fft.fft2(img[i])) for i in range(img.shape[0])])


    def show_original_image(self, unnormalize: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        """Show the original image with the help of matplotlib. Images
        can be unnormalized by providing a callable function.

        Parameters
        ----------
        Unnormalize: Optional[Callalbe[[np.ndarray], np.ndarray]] = None
            Function to unnormalize the image before showing
            it. Defaults to None. If no callable is provided, this
            argument is not used.
        """
        npimg = self.original_image.cpu().detach().numpy()
        if unnormalize:
            npimg = unnormalize(npimg)
            plt.imshow(np.transpose(npimg, (1,2,0)))
            plt.show()
        else:
            plt.imshow(np.transpose(npimg, (1,2,0)))
            plt.show()
            

    def show_fourier_image(self, unnormalize: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        """Show the fourier image with the help of matplotlib. Images
        can be unnormalized by providing a callable function.

        Parameters
        ----------
        Unnormalize: Optional[Callalbe[[np.ndarray], np.ndarray]] = None
            Function to unnormalize the image before showing
            it. Defaults to None. If no callable is provided, this
            argument is not used.
        """
        assert self.fourier_transform is not None
        npimg = self.fourier_transform.cpu().detach().numpy()
        if unnormalize:
            npimg = unnormalize(npimg)
            plt.imshow(np.transpose(npimg, (1,2,0)))
            plt.show()
        else:
            plt.imshow(np.transpose(npimg, (1,2,0)))
            plt.show()




@dataclass(init = False)
class BatchedImages(Image):
    """Class for the batched images. It contains a dictionary mapping
    the original images to their fourier transformations."""
    dim: Tuple[int, int, int, int]
    images: list[Image]

    def __init__(self, images: Sequence[Image]) -> None:
        self.original_images = torch.tensor(np.array(images))
        dimension = images[0].dim
        counter = 0
        for im in images:
            assert im.dim == dimension
            counter +=1
        self.dim = (counter, *dimension)
        assert isinstance(self.dim, tuple)

        assert self.original_image.shape == self.dim

    def fourier_transform(self, images: Sequence[Image]) -> None:
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
        assert len(images) >= 1 # Asserts the images are batched
        [img[i].fourier_transform_single_image(img[i]) for img in images]

