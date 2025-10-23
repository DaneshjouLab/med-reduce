import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, balanced_accuracy_score
)
import random
from torch.utils.data import Subset, DataLoader

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # BCEWithLogitsLoss expects raw logits

def train_and_validate(model, train_loader, val_loader, lr, num_epochs=10, device='cpu'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels, _, _ in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float().unsqueeze(1)

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
            inputs, labels = inputs.to(device).float(), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    return deepcopy(model), avg_val_loss

def tune_logistic_regression(train_loader, val_loader, input_dim, lr_list, num_epochs=10, device='cpu'):
    best_model = None
    best_val_loss = float('inf')
    best_lr = None

    losses = []
    for lr in lr_list:
        # print(f"Training with learning rate: {lr}")
        model = LogisticRegressionModel(input_dim)
        trained_model, val_loss = train_and_validate(model, train_loader, val_loader, lr, num_epochs, device)

        losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = trained_model
            best_lr = lr
        print(f"Learning rate: {lr:.4g}, Validation loss: {val_loss:.4f}, Best Validation loss: {best_val_loss:.4f}")

    # plt.figure(figsize=(6, 6))
    # plt.plot(lr_list, losses, color="green")
    # plt.xlabel('Learning Rate')
    # plt.xscale('log')
    # plt.ylabel('Validation Loss')
    # plt.grid(True)
    # plt.show()

    print(f"Best learning rate: {best_lr} with validation loss: {best_val_loss:.4f}")
    return best_model, losses

def train_logistic_regression(train_loader, val_loader, lr_list=np.logspace(np.log10(1e-6), np.log10(1e2), num=33)): 
    input_dim = next(iter(train_loader))[0].shape[1] 
    best_model, losses = tune_logistic_regression(
        train_loader, val_loader, input_dim=input_dim,
        lr_list=lr_list, num_epochs=20, device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return best_model, losses

def evaluate_model(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, labels, _, _ in test_loader:
            inputs = inputs.to(device).float()
            outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
            y_probs.extend(outputs)
            y_true.extend(labels.numpy())

    # Binary predictions (threshold at 0.5)
    y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

    # Metrics
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, y_probs)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    # Plot AUROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (AUROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Print metrics
    print("\n📊 Test Set Evaluation:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc_score:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc_score,
        "balanced_acc": balanced_acc,
        "fpr": fpr,
        "tpr": tpr,
        "y_true": y_true,
        "y_probs": y_probs,
        "y_pred": y_pred,
    }

def split_dataset_by_patient(dataset, batch_size, num_workers, seed=42, verbose=True):
    """
    Splits a dataset into train, val, and test subsets (70%, 10%, 20%),
    ensuring all samples from a patient are in only one subset.
    
    Assumes dataset[i] returns (data, label, patient_id).
    """
    random.seed(seed)

    # Step 1: Map patient_id to sample indices
    patient_to_indices = {}
    for idx, sample in enumerate(dataset):
        patient_id = sample[2]  # patient ID is at index 2
        patient_to_indices.setdefault(patient_id, []).append(idx)

    # Step 2: Shuffle and split patient IDs
    all_patients = list(patient_to_indices.keys())
    random.shuffle(all_patients)

    num_patients = len(all_patients)
    train_cutoff = int(0.7 * num_patients)
    val_cutoff = int(0.8 * num_patients)

    train_patients = set(all_patients[:train_cutoff])
    val_patients   = set(all_patients[train_cutoff:val_cutoff])
    test_patients  = set(all_patients[val_cutoff:])

    # Step 3: Ensure no overlap
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    # Step 4: Collect indices for each split
    train_indices = [i for p in train_patients for i in patient_to_indices[p]]
    val_indices   = [i for p in val_patients   for i in patient_to_indices[p]]
    test_indices  = [i for p in test_patients  for i in patient_to_indices[p]]

    # Step 5: Print sizes if verbose
    if verbose:
        total = len(dataset)
        print(f"Total samples: {total}")
        print(f"Training set size:   {len(train_indices)} ({round(len(train_indices)/total * 100, 1)}%)")
        print(f"Validation set size: {len(val_indices)} ({round(len(val_indices)/total * 100, 1)}%)")
        print(f"Testing set size:    {len(test_indices)} ({round(len(test_indices)/total * 100, 1)}%)")

        total_patients = len(all_patients)
        print(f"\nPatient counts — Total: {total_patients}, Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")

    # Step 6: Return subsets
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_indices, val_indices, test_indices