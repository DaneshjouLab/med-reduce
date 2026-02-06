import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import (
    AutoModel,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import ImageClassifierOutput

# ============================================================================
# Custom DINOv3 Classification Model Wrapper
# ============================================================================

class DINOv3Config(PretrainedConfig):
    """Config for DINOv3 classifier to make it compatible with HuggingFace"""
    model_type = "dinov3_classifier"
    
    def __init__(
        self,
        backbone_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_labels: int = 2,
        hidden_size: int = 384,
        dropout_rate: float = 0.1,
        use_quantization: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_model_id = backbone_model_id
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_quantization = use_quantization


class DINOv3ForImageClassification(PreTrainedModel):
    """
    DINOv3 model with classification head.
    Compatible with HuggingFace Trainer API.
    """
    config_class = DINOv3Config
    
    def __init__(self, config: DINOv3Config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        
        # Load backbone
        self.backbone = AutoModel.from_pretrained(
            config.backbone_model_id,
            torch_dtype=torch.float32
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        # Initialize weights
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """Initialize classifier weights"""
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
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass compatible with Trainer API.
        
        Args:
            pixel_values: Processed images
            labels: Ground truth labels
            return_dict: Whether to return ModelOutput
        
        Returns:
            ModelOutput with loss, logits, and optionally hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get features from backbone
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # Return in format expected by Trainer
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )