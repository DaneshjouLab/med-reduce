import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
import math


@dataclass
class SegmentationOutput(ModelOutput):
    """
    Args:
        loss: Optional training loss (Dice + BCE)
        logits: Segmentation logits of shape [batch_size, num_classes, H, W]
        hidden_states: Optional hidden states from backbone
        attentions: Optional attention weights from backbone
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


class DINOv3SegmentationConfig(PretrainedConfig):
    """Config for DINOv3 segmentation model."""
    model_type = "dinov3_segmentation"

    def __init__(
        self,
        backbone_model_id: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        num_classes: int = 1,  # Binary segmentation by default
        hidden_size: int = 384,
        patch_size: int = 16,
        image_size: int = 224,
        dropout_rate: float = 0.1,
        use_quantization: bool = False,
        loss_type: str = "dice_bce",  # "dice", "bce", or "dice_bce"
        dice_weight: float = 0.5,  # Weight for Dice loss in combined loss
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_model_id = backbone_model_id
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        self.use_quantization = use_quantization
        self.loss_type = loss_type
        self.dice_weight = dice_weight


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes, H, W]
            targets: [batch_size, num_classes, H, W] or [batch_size, H, W]

        Returns:
            Dice loss (scalar)
        """
        probs = torch.sigmoid(logits)

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)  # [B, 1, H, W]

        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class DINOv3ForSegmentation(PreTrainedModel):
    config_class = DINOv3SegmentationConfig

    def __init__(self, config: DINOv3SegmentationConfig):
        super().__init__(config)
        self.config = config
        self.num_classes = config.num_classes

        self.backbone = AutoModel.from_pretrained(
            config.backbone_model_id,
            torch_dtype=torch.float32
        )

        self.num_patches_per_side = config.image_size // config.patch_size

        self.seg_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.Conv2d(config.hidden_size, config.num_classes, kernel_size=1)
        )

        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self._init_weights(self.seg_head)

    def _init_weights(self, module):
        """Initialize head weights."""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Args:
            pixel_values: Processed images [batch_size, 3, H, W]
            labels: Ground truth masks [batch_size, H, W] or [batch_size, num_classes, H, W]
                    Values should be 0 or 1 for binary segmentation
            return_dict: Whether to return ModelOutput

        Returns:
            SegmentationOutput with loss, logits, and optionally hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.backbone(pixel_values=pixel_values)

        # Get patch tokens (exclude CLS token)
        # Shape: [batch_size, num_patches + 1, hidden_size]
        sequence_output = outputs.last_hidden_state

        # Remove CLS token (first token)
        patch_tokens = sequence_output[:, 1:, :]  # [B, num_patches, hidden_size]

        batch_size = patch_tokens.size(0)
        hidden_size = patch_tokens.size(2)

        # Reshape to spatial grid
        # [B, num_patches, hidden_size] -> [B, hidden_size, H_patches, W_patches]
        patch_tokens = patch_tokens.transpose(1, 2).contiguous()  # [B, hidden_size, num_patches]
        patch_tokens = patch_tokens.view(
            batch_size,
            hidden_size,
            self.num_patches_per_side,
            self.num_patches_per_side
        )

        # Apply segmentation head (1x1 conv)
        # [B, hidden_size, H_patches, W_patches] -> [B, num_classes, H_patches, W_patches]
        logits = self.seg_head(patch_tokens)

        # Upsample to original resolution
        # [B, num_classes, H_patches, W_patches] -> [B, num_classes, H, W]
        target_size = pixel_values.shape[2:]  # (H, W)
        logits = F.interpolate(
            logits,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        loss = None
        if labels is not None:
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            if labels.shape[2:] != logits.shape[2:]:
                labels = F.interpolate(
                    labels.float(),
                    size=logits.shape[2:],
                    mode='nearest'
                )

            if self.config.loss_type == "dice":
                loss = self.dice_loss(logits, labels)
            elif self.config.loss_type == "bce":
                loss = self.bce_loss(logits, labels.float())
            elif self.config.loss_type == "dice_bce":
                dice_loss = self.dice_loss(logits, labels)
                bce_loss = self.bce_loss(logits, labels.float())
                loss = self.config.dice_weight * dice_loss + (1 - self.config.dice_weight) * bce_loss
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SegmentationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
