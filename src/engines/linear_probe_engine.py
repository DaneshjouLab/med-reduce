"""Linear probe training engine for logistic regression."""
import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from typing import Tuple, List
from torch.utils.data import DataLoader

from src.wrappers.probe import LogisticRegressionModel
from src.losses.classification import get_classification_loss


def train_and_validate(
    model: LogisticRegressionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    num_epochs: int = 10,
    device: str = 'cpu'
) -> Tuple[LogisticRegressionModel, float]:
    """
    Train and validate logistic regression model.
    
    Args:
        model: Logistic regression model
        train_loader: Training data loader
        val_loader: Validation data loader
        lr: Learning rate
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        Tuple of (trained_model, validation_loss)
    """
    model = model.to(device)
    criterion = get_classification_loss("bce")
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels, _, _ in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels, _, _ in val_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    return deepcopy(model), avg_val_loss


def tune_logistic_regression(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    lr_list: List[float],
    num_epochs: int = 10,
    device: str = 'cpu'
) -> Tuple[LogisticRegressionModel, List[float]]:
    """
    Tune logistic regression by searching over learning rates.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input feature dimension
        lr_list: List of learning rates to try
        num_epochs: Number of epochs per trial
        device: Device to train on
        
    Returns:
        Tuple of (best_model, validation_losses)
    """
    best_model = None
    best_val_loss = float('inf')
    best_lr = None

    losses = []
    for lr in lr_list:
        model = LogisticRegressionModel(input_dim)
        trained_model, val_loss = train_and_validate(
            model, train_loader, val_loader, lr, num_epochs, device
        )

        losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = trained_model
            best_lr = lr
            
        print(
            f"Learning rate: {lr:.4g}, "
            f"Validation loss: {val_loss:.4f}, "
            f"Best Validation loss: {best_val_loss:.4f}"
        )

    print(f"Best learning rate: {best_lr} with validation loss: {best_val_loss:.4f}")
    return best_model, losses


def train_logistic_regression(
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr_list: np.ndarray = None
) -> Tuple[LogisticRegressionModel, List[float]]:
    """
    Train logistic regression with automatic hyperparameter tuning.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        lr_list: Optional list of learning rates to try
        
    Returns:
        Tuple of (best_model, validation_losses)
    """
    if lr_list is None:
        lr_list = np.logspace(np.log10(1e-6), np.log10(1e2), num=33)
    
    input_dim = next(iter(train_loader))[0].shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    best_model, losses = tune_logistic_regression(
        train_loader, val_loader,
        input_dim=input_dim,
        lr_list=lr_list,
        num_epochs=20,
        device=device
    )
    
    return best_model, losses

