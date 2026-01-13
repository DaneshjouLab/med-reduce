import torch
import torch.nn as nn
from typing import Optional
from transformers import (
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class FeatureDetectionOutput(ModelOutput):
    """
    Output for multi-label feature detection.

    Args:
        loss: Optional training loss (BCE with logits)
        logits: Multi-label logits of shape [batch_size, num_features]
        hidden_states: Optional hidden states from backbone
        attentions: Optional attention weights from backbone
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


class DINOv3FeatureDetectionConfig(PretrainedConfig):
    """Config for DINOv3 dermoscopy feature detection model."""
    model_type = "dinov3_feature_detection"

    def __init__(
        self,
        backbone_model_id: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        num_features: int = 4,
        hidden_size: int = 384,
        dropout_rate: float = 0.1,
        use_quantization: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_model_id = backbone_model_id
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_quantization = use_quantization


class DINOv3ForFeatureDetection(PreTrainedModel):
    """
    DINOv3 model for dermoscopy feature detection (multi-label classification).

    Architecture:
        - Global pooled embedding from DINOv3 backbone
        - Linear projection to feature space
        - Multi-label logits (one per feature)

    Features detected (for ISIC):
        - Pigment Network
        - Negative Network
        - Streaks
        - Milia-like Cysts

    Loss: Binary Cross Entropy with Logits (BCEWithLogitsLoss)
    """
    config_class = DINOv3FeatureDetectionConfig

    def __init__(self, config: DINOv3FeatureDetectionConfig):
        super().__init__(config)
        self.config = config
        self.num_features = config.num_features

        # Load backbone
        self.backbone = AutoModel.from_pretrained(
            config.backbone_model_id,
            torch_dtype=torch.float32
        )

        # Feature detection head (multi-label)
        self.feature_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.num_features)
        )

        # Initialize weights
        self._init_weights(self.feature_head)

    def _init_weights(self, module):
        """Initialize head weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass for feature detection.

        Args:
            pixel_values: Processed images [batch_size, 3, H, W]
            labels: Ground truth multi-label features [batch_size, num_features]
                    Values should be 0 or 1 for each feature
            return_dict: Whether to return ModelOutput

        Returns:
            FeatureDetectionOutput with loss, logits, and optionally hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get features from backbone
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output

        # Multi-label feature prediction
        logits = self.feature_head(pooled_output)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # BCEWithLogitsLoss for multi-label classification
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return FeatureDetectionOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
