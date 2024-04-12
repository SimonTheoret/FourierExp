from transformers import (
    MobileViTConfig,
    MobileViTForImageClassification,
)
import torch.nn as nn
import torch
from typing import Optional, Tuple


class MobileViT(nn.Module):
    def __init__(
        self,
        configuration: Optional[MobileViTConfig] = None,
        from_pretrained: bool = False,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        # Configuration. Could probably add options around here.
        if configuration is None:
            self.configuration: MobileViTConfig = MobileViTConfig(
                num_channels=3,
                image_size=32,
                num_labels=10,
                hidden_size=[64, 80, 96],
                neck_hidden_sizes=[16, 16, 24, 48, 64, 80, 320],
                output_stride = 16,
            )
        else:
            self.configuration: MobileViTConfig = configuration

        # Model loading, possibly from a pretrained version.
        if from_pretrained:
            out = MobileViTForImageClassification.from_pretrained(
                "apple/mobilevit-small"
            )
            if isinstance(out, MobileViTForImageClassification):
                self.model = out
            else:
                self.model = MobileViTForImageClassification(self.configuration)
        else:
            self.model = MobileViTForImageClassification(self.configuration)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x, return_dict=False)[0]
        assert isinstance(logits, torch.Tensor)
        return logits

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()
