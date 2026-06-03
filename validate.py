"""
Utility script to validate data loading and model initialization.
Run this first to ensure everything is set up correctly.
"""

import os
import sys
import yaml
import torch
from transformers import AutoTokenizer

# Import custom modules
from src.data import get_dataloaders, HomonymDataset
from src.model import create_model


def validate_environment():
    """Check if all required files and dependencies are available."""
    print("="*60)
    print("Environment Validation")
    print("="*60)
    
    # Check config
    if not os.path.exists("config.yaml"):
        print("✗ config.yaml not found!")
        return False
    print("✓ config.yaml found")
    
    # Check data files
    if not os.path.exists("train.json"):
        print("✗ train.json not found!")
        return False
    print("✓ train.json found")
    
    if not os.path.exists("dev.json"):
        print("✗ dev.json not found!")
        return False
    print("✓ dev.json found")
    
    # Check Python packages
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__} installed")
    except ImportError:
        print("✗ Transformers not installed")
        return False
    
    try:
        import yaml
        print("✓ PyYAML installed")
    except ImportError:
        print("✗ PyYAML not installed")
        return False
    
    try:
        import scipy
        print("✓ SciPy installed")
    except ImportError:
        print("✗ SciPy not installed")
        return False
    
    return True


def validate_data(config):
    """Load and inspect data."""
    print("\n" + "="*60)
    print("Data Validation")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = HomonymDataset(
        config['train_file'],
        tokenizer,
        max_length=config['max_length'],
    )
    
    dev_dataset = HomonymDataset(
        config['dev_file'],
        tokenizer,
        max_length=config['max_length'],
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Dev dataset: {len(dev_dataset)} samples")
    
    # Inspect sample
    print("\nInspecting first training sample...")
    sample = train_dataset[0]
    print(f"  Sample keys: {sample.keys()}")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Target (normalized): {sample['target'].item():.4f}")
    print(f"  Target (original): {sample['average'].item():.4f}")
    print(f"  Stdev: {sample['stdev'].item():.4f}")
    print(f"  Sample ID: {sample['sample_id']}")
    
    # Check target distribution
    import torch
    targets = [train_dataset[i]['average'].item() for i in range(min(50, len(train_dataset)))]
    print(f"\nTarget score statistics (first 50 samples):")
    print(f"  Mean: {torch.tensor(targets).mean():.2f}")
    print(f"  Std: {torch.tensor(targets).std():.2f}")
    print(f"  Min: {min(targets):.2f}")
    print(f"  Max: {max(targets):.2f}")
    
    return True


def validate_model(config):
    """Initialize and test model."""
    print("\n" + "="*60)
    print("Model Validation")
    print("="*60)
    
    print("\nInitializing model...")
    model = create_model(config)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_length = config['max_length']
    
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    model = model.to(device)
    
    input_ids = torch.randint(0, 28996, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    token_type_ids = torch.zeros(batch_size, seq_length).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    
    if outputs.shape != (batch_size,):
        print(f"✗ Unexpected output shape: {outputs.shape}")
        return False
    
    if torch.isnan(outputs).any():
        print("✗ NaN detected in output!")
        return False
    
    print("✓ Model forward pass successful")
    return True


def validate_dataloaders(config):
    """Test DataLoader functionality."""
    print("\n" + "="*60)
    print("DataLoader Validation")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    print("\nCreating DataLoaders...")
    train_loader, dev_loader = get_dataloaders(
        config['train_file'],
        config['dev_file'],
        tokenizer,
        batch_size=config['batch_size'],
        eval_batch_size=config['eval_batch_size'],
        max_length=config['max_length'],
    )
    
    print(f"✓ Train DataLoader: {len(train_loader)} batches")
    print(f"✓ Dev DataLoader: {len(dev_loader)} batches")
    
    # Get a batch
    print("\nLoading first training batch...")
    batch = next(iter(train_loader))
    
    print(f"  Batch keys: {batch.keys()}")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  target shape: {batch['target'].shape}")
    print(f"  stdev shape: {batch['stdev'].shape}")
    
    if batch['target'].shape[0] != config['batch_size']:
        print(f"✗ Unexpected batch size: {batch['target'].shape[0]}")
        return False
    
    print("✓ DataLoader working correctly")
    return True


def main():
    """Run all validations."""
    print("\n" + "="*70)
    print("HOMONYM SENSE PLAUSIBILITY PREDICTION - VALIDATION SUITE")
    print("="*70)
    
    # Check environment
    if not validate_environment():
        print("\n✗ Environment validation failed!")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Load config
    config = yaml.safe_load(open("config.yaml"))
    
    # Validate data
    if not validate_data(config):
        print("\n✗ Data validation failed!")
        return False
    
    # Validate model
    if not validate_model(config):
        print("\n✗ Model validation failed!")
        return False
    
    # Validate DataLoaders
    if not validate_dataloaders(config):
        print("\n✗ DataLoader validation failed!")
        return False
    
    print("\n" + "="*70)
    print("✓ ALL VALIDATIONS PASSED!")
    print("="*70)
    print("\nYou can now run training:")
    print("  python src/train.py config.yaml")
    print("\nOr interactive validation:")
    print("  python -m src.data")
    print("  python -m src.model")
    print("  python -m src.metrics")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
