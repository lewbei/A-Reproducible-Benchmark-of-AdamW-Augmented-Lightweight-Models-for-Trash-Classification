"""
lightweight_model.py · PyTorch ≥ 2.1 · timm ≥ 0.9
---------------------------------------------------------------------------
Four selected lightweight backbones + a generic classifier wrapper.
Models selected based on parameter efficiency and accuracy:
- MobileNet V3-Large
- ViT-Small
- EfficientFormer L1
- ShuffleNet V2-x1.0
"""

import torch
import torch.nn as nn
import timm                      # provides ViT, EfficientFormer, MobileNet
from torchvision.models import (
    shufflenet_v2_x1_0,
    ShuffleNet_V2_X1_0_Weights,
)


# ╭──────────────────────────── backbone factories ───────────────────────────╮
def mobilenet_v3_large_backbone(*, in_chans: int = 3) -> nn.Module:
    """MobileNet V3-Large · Efficient mobile architecture."""
    return timm.create_model(
        "mobilenetv3_large_100",
        pretrained=False,
        in_chans=in_chans,
        num_classes=0,
        global_pool="avg",
    )

def vit_small_backbone(*, in_chans: int = 3) -> nn.Module:
    """ViT-Small · Vision Transformer with patch size 16."""
    return timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        in_chans=in_chans,
        num_classes=0,
        global_pool="token",
    )


def efficientformer_l1_backbone(*, in_chans: int = 3) -> nn.Module:
    """EfficientFormer-L1 · Hybrid CNN-Transformer architecture."""
    return timm.create_model(
        "efficientformer_l1",
        pretrained=False,
        in_chans=in_chans,
        num_classes=0,
        global_pool="avg",
    )


def shufflenetv2_x1_backbone(*, in_chans: int = 3) -> nn.Module:
    """ShuffleNet V2-x1.0 · Ultra-efficient architecture."""
    m = shufflenet_v2_x1_0(weights=None)
    if in_chans != 3:
        m.conv1[0] = nn.Conv2d(in_chans, 24, kernel_size=3, stride=2, padding=1, bias=False)
    return m
# ╰───────────────────────────────────────────────────────────────────────────╯


class GenericClassifier(nn.Module):
    """
    Wrap *any* backbone that ends with global pooling + dense head.
    Removes the backbone's head and appends a new `nn.Linear`.
    """

    def __init__(self, num_classes: int, base_model_fn, **backbone_kwargs):
        super().__init__()

        # Build backbone & remember its factory name
        self.base = base_model_fn(**backbone_kwargs)
        self.base_model_name = base_model_fn.__name__

        # ---- Strip original classifier -------------------------------------------------
        if hasattr(self.base, "fc"):          # some CNNs
            self.base.fc = nn.Identity()
        elif hasattr(self.base, "classifier"):
            self.base.classifier = nn.Identity()
        elif hasattr(self.base, "head"):      # most ViT/timm nets
            self.base.head = nn.Identity()
        else:
            raise AttributeError(
                f"{self.base_model_name} exposes no `.fc`, `.classifier`, or `.head`."
            )

        # Determine feature dimension
        device = next(self.base.parameters()).device
        dummy = torch.randn(1, backbone_kwargs.get("in_chans", 3), 224, 224, device=device)
        with torch.no_grad():
            features = self.base(dummy)
            if features.dim() > 2:  # Handle cases where features aren't flattened
                features = features.mean([2, 3])  # Global average pooling
            feat_dim = features.shape[1]
            print(f"{self.base_model_name:30s} feature dim: {feat_dim}")

        # New task-specific head
        self.classifier = nn.Linear(feat_dim, num_classes)

    # ---------------------------------------------------------------------
    def forward(self, x):
        feats = self.base(x)               # (B, C)   or   (B, C, H, W)

        # --- make sure feats is flat ------------------------------------
        if feats.dim() == 4:               # CNN feature map
            feats = feats.mean([2, 3])     # global average pool -> (B, C)
        elif feats.dim() == 3:             # ViT tokens (B, N, C)
            feats = feats.mean(1)          # token average   -> (B, C)
        # ----------------------------------------------------------------

        return self.classifier(feats)


# ╭────────────────────────── quick sanity test  (optional) ───────────────────╮
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fn in (
        mobilenet_v3_large_backbone,
        vit_small_backbone,
        efficientformer_l1_backbone,
        shufflenetv2_x1_backbone
    ):
        model = GenericClassifier(
            num_classes=6,        # updated for your dataset
            base_model_fn=fn,
            in_chans=3,
        ).to(device)
        out = model(torch.randn(2, 3, 224, 224).to(device))
        print(f"{model.base_model_name:30s} -> {tuple(out.shape)}")
        
        # Print model size info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{model.base_model_name:30s} total params: {total_params:,}")
