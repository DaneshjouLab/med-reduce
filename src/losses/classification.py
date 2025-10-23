"""Classification loss functions."""
import torch.nn as nn


def get_classification_loss(loss_type: str = "bce"):
    """
    Get classification loss function.
    
    Args:
        loss_type: Type of loss ('bce' for binary cross entropy)
        
    Returns:
        Loss function
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

