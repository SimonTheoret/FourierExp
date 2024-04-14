from dataclasses import dataclass
from typing import Callable, Optional
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch


# TODO: This class is not useful anymore. The right way to proceed
# would be to only use BatchedImages to compute the fft2 and the
# fftshift.


@dataclass(init=False)
class Image:
    """
    Class for a single image. Contains the dimension of the image,
    the original image and optionaly the fourier transform of the
    original image.
    """

    dim: tuple[int, int, int]
    original_image: torch.Tensor
    fourier_transform: Optional[torch.Tensor]
    fourier_high_pass: Optional[torch.Tensor]
    fourier_low_pass: Optional[torch.Tensor]

    def __init__(self, image: npt.ArrayLike, dim: tuple[int, int, int]) -> None:
        """Builds a Image object"""
        if not isinstance(image, torch.Tensor):
            self.original_image = torch.tensor(np.array(image))
        else:
            self.original_image = image
        self.dim = dim
        assert tuple(self.original_image.shape) == self.dim
        assert (
            self.original_image.shape[0] == 3
        )  # Make sure the channels are at the right position

    def fourier_transform_single_image(self) -> None:
        """
        Utility function for updating the fourier transform on a
        single image with C channels. It applies the 2D FFT algorithm
        and then shifts the (recenter) the transform.

        Parameters
        ----------
        img: torch.Tensor
            Tensor of dimension (CxHxW). This must be a single image,
            not a batched image.

        Returns the *shifted* fourier transform of the image.
        """
        assert len(self.original_image) == 3  # Asserts it contains a channel
        self.fourier_transform = torch.tensor(
            [
                torch.fft.fftshift(torch.fft.fft2(self.original_image[i]), dim=(1, 2))
                for i in range(self.original_image.shape[0])
            ]
        )

    def show_original_image(
        self, unnormalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> None:
        """
        Show the original image with the help of matplotlib. Images
        can be unnormalized by providing a callable function.

        Parameters
        ----------
        Unnormalize: Optional[Callalbe[[np.ndarray], np.ndarray]] =
        None
            Function to unnormalize the image before showing
            it. Defaults to None. If no callable is provided, this
            argument is not used.
        """
        npimg = self.original_image.cpu().detach().numpy()
        if unnormalize:
            npimg = unnormalize(npimg)
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

    def show_fourier_image(
        self, unnormalize: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> None:
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
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

    def to_np(self) -> np.ndarray:
        """
        Returns the original_image to a numpy array
        """
        return self.original_image.cpu().detach().numpy()

    # @deprecated  # Use BatchedImages instead
    # def filter_high_pass(self, square_side_length: int) -> None:
    # """
    # Filters high frequency content and updates the
    # fourier_high_pass attributes. It uses a square with sides of
    # length `square_side_lenghts` to mask the high frequency
    # content.

    # Parameters
    # ----------
    # square_side_length: int
    #     Length of a side in the square located in the center of the image.

    # """
    # assert self.fourier_transform is not None
    # b, h, w = self.dim  # height and width
    # high_freq_centered = torch.zeros_like(self.fourier_transform)
    # for i in range(b):  # shift the frequencies to the center of the image
    #     high_freq_centered[i, :, :] = torch.fft.ifftshift(
    #         self.fourier_transform.clone()
    #     )
    # for i in range(b):
    #     cy, cx = int(h // 2 + 1), int(w // 2 + 1)  # centerness, should be (17, 17)
    # pass


@dataclass(init=False)
class BatchedImages(Image):
    """
    Class for the batched images. It contains a dictionary mapping the
    original images to their fourier transformations.
    """

    dim: tuple[int, int, int, int]
    images: list[Image]
    images_tensor: Optional[torch.Tensor]
    fourier_tensor: Optional[torch.Tensor]
    high_pass_fourier: Optional[torch.Tensor]
    low_pass_fourier: Optional[torch.Tensor]

    def __init__(self, images: Sequence[Image], colored: bool = True) -> None:
        dimension = images[0].dim
        for im in images:
            assert im.dim == dimension
        self.dim = (len(images), *dimension)
        assert isinstance(self.dim, tuple)
        self.images_tensor = torch.cat(
            [img.original_image.unsqueeze(0) for img in images]
        )
        assert isinstance(self.images_tensor, torch.Tensor)
        if colored:
            assert len(self.images_tensor.shape) == 4
        else:
            assert len(self.images_tensor.shape) == 3
        assert tuple(list(self.images_tensor.shape)) == self.dim

    def fourier_transform_all(self) -> None:
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
        assert self.images_tensor is not None
        assert (
            len(list(self.images_tensor.shape)) >= 1
        )  # Asserts the images are batched
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # uses the gpu if possible
        self.fourier_tensor = torch.fft.fftshift(
            torch.fft.fft2(self.images_tensor.to(device)), dim=(2, 3)
        )

    def to_np(self) -> np.ndarray:
        """
        Returns a numpy array containing all the images.
        """
        assert self.images_tensor is not None
        array = self.images_tensor.detach().cpu().numpy()
        assert isinstance(array, np.ndarray)
        return array

    def filter_high_pass(self, square_side_length: int) -> None:
        """
        Filters high frequency content and updates the
        fourier_high_pass attributes. It uses a square with sides of
        length `square_side_lenghts` to mask the high frequency
        content.

        Parameters
        ----------
        square_side_length: int
            Length of a side in the square located in the center of
            the image.

        """
        assert self.fourier_tensor is not None
        high_freq_centered = (
            torch.fft.ifftshift(  # shifts the HF in the center of the image
                self.fourier_tensor.clone(), dim=(2, 3)
            )
        )  # Only shift the height and width dimensions
        assert isinstance(high_freq_centered, torch.Tensor)
        h, w = high_freq_centered.size(dim=2), high_freq_centered.size(dim=3)
        ch, cw = h // 2, w // 2
        half_length = square_side_length // 2
        high_freq_centered[
            :,
            :,
            ch - half_length : ch + half_length,
            cw - half_length : cw + half_length,
        ] = 0
        # TODO: recompose the original but filtered image with torch.fft
        # TODO: Do the same for the low pass
        self.high_pass_fourier = torch.fft.ifft2(high_freq_centered).real

    def filter_low_pass(self, square_side_length: int) -> None:
        """
        Filters low frequency content and updates the
        fourier_low_pass attributes. It uses a square with sides of
        length `square_side_lenghts` to mask the high frequency
        content.

        Parameters
        ----------
        square_side_length: int
            Length of a side in the square located in the center of
            the image.

        """
        assert self.fourier_tensor is not None
        low_freq_centered = (
            self.fourier_tensor.clone()
        )  # No need to shift, LF are already centered
        assert isinstance(low_freq_centered, torch.Tensor)
        h, w = low_freq_centered.size(dim=2), low_freq_centered.size(dim=3)
        ch, cw = h // 2, w // 2
        half_length = square_side_length // 2
        low_freq_centered[
            :,
            :,
            ch - half_length : ch + half_length,
            cw - half_length : cw + half_length,
        ] = 0
        self.low_pass_fourier = torch.fft.ifft2(
            torch.fft.ifftshift(low_freq_centered, dim=(2, 3))
        ).real

    def show_image(self, img: torch.Tensor) -> None:
        """Shows an image"""
        img = img / 2 + 0.5  # unnormalize
        npimg = img.cpu().detach().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
