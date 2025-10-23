"""Data module for splitting datasets and creating data loaders."""
import random
from typing import Tuple, List
from torch.utils.data import Subset, DataLoader

from src.config import DataSplitConfig


def split_dataset_by_patient(
    dataset,
    config: DataSplitConfig,
    batch_size: int,
    num_workers: int,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int], List[int], List[int]]:
    """
    Split dataset into train, val, and test subsets by patient.
    
    Ensures all samples from a patient are in only one subset.
    Assumes dataset[i] returns (data, label, patient_id, ...).
    
    Args:
        dataset: PyTorch Dataset
        config: Data split configuration
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loaders
        verbose: Whether to print split information
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, 
                 train_indices, val_indices, test_indices)
    """
    random.seed(config.seed)

    # Step 1: Map patient_id to sample indices
    patient_to_indices = {}
    for idx, sample in enumerate(dataset):
        patient_id = sample[2]  # patient ID is at index 2
        patient_to_indices.setdefault(patient_id, []).append(idx)

    # Step 2: Shuffle and split patient IDs
    all_patients = list(patient_to_indices.keys())
    random.shuffle(all_patients)

    num_patients = len(all_patients)
    train_cutoff = int(config.train_ratio * num_patients)
    val_cutoff = int((config.train_ratio + config.val_ratio) * num_patients)

    train_patients = set(all_patients[:train_cutoff])
    val_patients = set(all_patients[train_cutoff:val_cutoff])
    test_patients = set(all_patients[val_cutoff:])

    # Step 3: Ensure no overlap
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    # Step 4: Collect indices for each split
    train_indices = [i for p in train_patients for i in patient_to_indices[p]]
    val_indices = [i for p in val_patients for i in patient_to_indices[p]]
    test_indices = [i for p in test_patients for i in patient_to_indices[p]]

    # Step 5: Print sizes if verbose
    if verbose:
        total = len(dataset)
        print(f"Total samples: {total}")
        print(
            f"Training set size:   {len(train_indices)} "
            f"({round(len(train_indices)/total * 100, 1)}%)"
        )
        print(
            f"Validation set size: {len(val_indices)} "
            f"({round(len(val_indices)/total * 100, 1)}%)"
        )
        print(
            f"Testing set size:    {len(test_indices)} "
            f"({round(len(test_indices)/total * 100, 1)}%)"
        )

        print(
            f"\nPatient counts — Total: {num_patients}, "
            f"Train: {len(train_patients)}, "
            f"Val: {len(val_patients)}, "
            f"Test: {len(test_patients)}"
        )

    # Step 6: Create data loaders
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    return (
        train_loader, val_loader, test_loader,
        train_indices, val_indices, test_indices
    )

