"""Linear probe (logistic regression) wrapper for classification."""
import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    """
    Simple logistic regression model for binary classification.
    
    Args:
        input_dim: Dimension of input features
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, 1)
        """
        return self.linear(x)  # BCEWithLogitsLoss expects raw logits

