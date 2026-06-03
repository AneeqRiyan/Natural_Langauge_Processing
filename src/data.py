"""
Data loading and preprocessing for homonym sense plausibility prediction.
Handles JSON loading, tokenization, and normalization of targets.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
import os


class HomonymDataset(Dataset):
    """
    PyTorch Dataset for homonym sense plausibility prediction.
    
    Concatenates context and meaning definition for DeBERTa input.
    Normalizes target scores to [0, 1] range for training.
    """
    
    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_length: int = 512,
        target_min: float = 1.0,
        target_max: float = 5.0,
        normalize: bool = True
    ):
        """
        Args:
            json_path: Path to JSON file with data records
            tokenizer: Hugging Face tokenizer instance
            max_length: Maximum sequence length for tokenization
            target_min: Minimum value of target scores
            target_max: Maximum value of target scores
            normalize: Whether to normalize targets to [0, 1]
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.normalize = normalize
        self.target_min = target_min
        self.target_max = target_max
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        # Convert dict to list of records preserving order
        self.records = [data_dict[str(i)] for i in range(len(data_dict))]
        
        print(f"Loaded {len(self.records)} records from {json_path}")
        
        # Validate data integrity
        self._validate_data()
    
    def _validate_data(self):
        """Check for missing required fields and valid values."""
        required_fields = ['precontext', 'sentence', 'judged_meaning', 'average', 'stdev']
        
        for i, record in enumerate(self.records):
            for field in required_fields:
                if field not in record:
                    raise ValueError(f"Record {i} missing field: {field}")
            
            # Validate score range
            avg = record.get('average')
            if not (self.target_min <= avg <= self.target_max):
                raise ValueError(f"Record {i} has out-of-range average: {avg}")
            
            # Validate stdev (should be non-negative, but allow 0)
            stdev = record.get('stdev', 0)
            if stdev < 0:
                raise ValueError(f"Record {i} has negative stdev: {stdev}")
    
    def _normalize_target(self, value: float) -> float:
        """Normalize target from [target_min, target_max] to [0, 1]."""
        if self.normalize:
            return (value - self.target_min) / (self.target_max - self.target_min)
        return value
    
    def _denormalize_target(self, value: float) -> float:
        """Denormalize target from [0, 1] to [target_min, target_max]."""
        if self.normalize:
            return value * (self.target_max - self.target_min) + self.target_min
        return value
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dict with keys: input_ids, attention_mask, token_type_ids, target,
                           sample_id, average, stdev (normalized)
        """
        record = self.records[idx]
        
        # Construct input: context + [SEP] + meaning definition
        precontext = record.get('precontext', '')
        sentence = record.get('sentence', '')
        ending = record.get('ending', '')
        meaning = record.get('judged_meaning', '')
        
        # Combine all context elements
        context = f"{precontext} {sentence}"
        if ending:
            context += f" {ending}"
        
        # Format: context [SEP] meaning_definition
        input_text = f"{context} [SEP] {meaning}"
        
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get target score
        target = float(record.get('average', 0))
        target_normalized = self._normalize_target(target)
        
        # Get auxiliary information
        sample_id = record.get('sample_id', str(idx))
        stdev = float(record.get('stdev', 0.5))
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'token_type_ids': encoded.get('token_type_ids', torch.zeros_like(encoded['input_ids'])).squeeze(),
            'target': torch.tensor(target_normalized, dtype=torch.float32),
            'sample_id': sample_id,
            'average': torch.tensor(target, dtype=torch.float32),  # Unnormalized for metrics
            'stdev': torch.tensor(stdev, dtype=torch.float32),
        }


def get_dataloaders(
    train_path: str,
    dev_path: str,
    tokenizer,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        train_path: Path to training JSON file
        dev_path: Path to development JSON file
        tokenizer: Hugging Face tokenizer
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        num_workers: Number of DataLoader workers
    
    Returns:
        Tuple of (train_loader, dev_loader)
    """
    train_dataset = HomonymDataset(
        train_path,
        tokenizer,
        max_length=max_length,
        normalize=True
    )
    
    dev_dataset = HomonymDataset(
        dev_path,
        tokenizer,
        max_length=max_length,
        normalize=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, dev_loader


if __name__ == "__main__":
    """Quick validation test."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    
    # Test dataset loading
    print("Testing dataset loading...")
    dataset = HomonymDataset(
        "train.json",
        tokenizer,
        max_length=512
    )
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Target (normalized): {sample['target'].item():.4f}")
    print(f"Target (original): {sample['average'].item():.4f}")
    print(f"Sample ID: {sample['sample_id']}")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    train_loader, dev_loader = get_dataloaders(
        "train.json",
        "dev.json",
        tokenizer,
        batch_size=8,
        eval_batch_size=16
    )
    
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch targets shape: {batch['target'].shape}")
    print(f"Batch stdev shape: {batch['stdev'].shape}")
    print(" Data loading validation complete!")
