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
        from_pretrained: bool = True,
    ):
        # Configuration. Could probably add options around here.
        if configuration is None:
            self.configuration: MobileViTConfig = MobileViTConfig()
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, logits, _ = self.model(
            x,
            return_dict=False,
        )
        # This is the main branch.
        if isinstance(loss, torch.Tensor) and isinstance(logits, torch.Tensor):
            return loss, logits
        # This branch should not happen and is only there to make sure the LSP is happy.
        else:
            assert isinstance(loss, torch.Tensor) and isinstance(logits, torch.Tensor)
            return torch.Tensor(0.0), torch.Tensor(0.0)
