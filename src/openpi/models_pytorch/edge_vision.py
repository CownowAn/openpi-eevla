"""EE-VLA edge view encoders for the PyTorch pi05 model.

Each edge view gets its own trainable DINOv3 ViT and gives one token: CLS -> MLP (dim -> width -> GELU ->
width) -> LayerNorm, where width is the hidden size of the transformer that reads the token (PaliGemma
for prefix placements, the action expert for suffix placements). Training-time edge dropout removes all
edge tokens of a sample with probability `dropout_p`. Ported from GR00T N1.7's EdgeVisionEncoder.
"""

import timm
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class EdgeVisionEncoder(nn.Module):
    def __init__(
        self,
        num_views: int,
        out_dim: int,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        dropout_p: float = 0.2,
        *,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_views, self.model_name, self.dropout_p = num_views, model_name, dropout_p
        # Built without weights: `load_pretrained_encoders()` loads DINOv3 when the checkpoint being
        # loaded has no edge encoders.
        self.encoders = nn.ModuleList(
            [timm.create_model(model_name, pretrained=False, num_classes=0) for _ in range(num_views)]
        )
        enc = self.encoders[0]
        dim = enc.num_features
        cfg = enc.pretrained_cfg
        self.image_size = tuple(cfg.get("input_size", (3, 256, 256))[1:])
        self.register_buffer(
            "mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1), persistent=False
        )
        self.proj = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim))
                for _ in range(num_views)
            ]
        )
        if grad_checkpointing:
            for e in self.encoders:
                e.set_grad_checkpointing(True)

    def load_pretrained_encoders(self):
        """Load the pretrained timm weights into every encoder (in place)."""
        state = timm.create_model(self.model_name, pretrained=True, num_classes=0).state_dict()
        for enc in self.encoders:
            enc.load_state_dict(state)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: [B, V, H, W, 3] uint8 -> edge tokens [B, V, out_dim]."""
        b, v = pixels.shape[:2]
        assert v == self.num_views, f"expected {self.num_views} edge views, got {v}"
        dtype = next(self.encoders[0].parameters()).dtype
        x = pixels.flatten(0, 1).permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False, antialias=True)
        x = ((x - self.mean) / self.std).unflatten(0, (b, v))
        tokens = [self.proj[i](self.encoders[i].forward_features(x[:, i].to(dtype))[:, :1]) for i in range(v)]
        return torch.cat(tokens, dim=1)

    def tokens_and_keep(self, pixels: torch.Tensor, view_masks: torch.Tensor, dtype: torch.dtype):
        """Edge tokens [B, V, D] and their keep mask [B, V]: the view masks with training-time dropout."""
        tokens = self.encode(pixels).to(dtype)
        keep = view_masks.to(torch.bool)
        if self.training and self.dropout_p > 0:
            keep = keep & (torch.rand(keep.shape[0], 1, device=keep.device) >= self.dropout_p)
        return tokens, keep
