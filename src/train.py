"""
Training script for homonym sense plausibility prediction model.
Implements mixed precision training, gradient accumulation, and validation.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict, Tuple
import time
import json

# Import custom modules
from data import get_dataloaders, HomonymDataset
from model import create_model
from metrics import compute_regression_metrics, format_metrics


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are proper types
    config['batch_size'] = int(config.get('batch_size', 16))
    config['gradient_accumulation_steps'] = int(config.get('gradient_accumulation_steps', 4))
    config['num_epochs'] = int(config.get('num_epochs', 3))
    config['learning_rate'] = float(config.get('learning_rate', 2e-5))
    config['weight_decay'] = float(config.get('weight_decay', 1e-2))
    config['warmup_ratio'] = float(config.get('warmup_ratio', 0.1))
    config['max_length'] = int(config.get('max_length', 512))
    config['eval_batch_size'] = int(config.get('eval_batch_size', 32))
    
    return config


def setup_training(config: Dict) -> Tuple[str, int]:
    """Setup training environment (directories, random seeds)."""
    # Create checkpoint and log directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Set random seeds for reproducibility
    seed = config['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Determine device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scaler: GradScaler,
    config: Dict,
    device: str,
    epoch: int,
    global_step: int,
) -> Tuple[float, int]:
    """
    Train for one epoch.
    
    Returns:
        (average_loss, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    grad_accumulation_steps = config['gradient_accumulation_steps']
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        total=len(train_loader),
        leave=False
    )
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config['mixed_precision']):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = nn.MSELoss()(logits, targets)
            # Scale loss for gradient accumulation
            loss = loss / grad_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation and optimizer step
        if (step + 1) % grad_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_step += 1
        
        total_loss += loss.item() * grad_accumulation_steps
        progress_bar.set_postfix({'loss': f"{loss.item() * grad_accumulation_steps:.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, global_step


def validate(
    model: nn.Module,
    dev_loader,
    device: str,
    config: Dict,
) -> Dict[str, float]:
    """
    Evaluate model on development set.
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_stdevs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            targets = batch['target'].to(device)
            stdevs = batch['stdev'].to(device)
            
            # Forward pass
            with autocast(enabled=config['mixed_precision']):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = nn.MSELoss()(logits, targets)
            
            total_loss += loss.item()
            
            all_predictions.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_stdevs.append(stdevs.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    stdevs = torch.cat(all_stdevs)
    
    # Compute metrics
    avg_loss = total_loss / len(dev_loader)
    metrics = compute_regression_metrics(predictions, targets, stdevs)
    metrics['loss'] = avg_loss
    
    return metrics


def train(config: Dict):
    """Main training loop."""
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    # Setup
    device = setup_training(config)
    
    # Load data and model
    print("\nLoading data...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    train_loader, dev_loader = get_dataloaders(
        config['train_file'],
        config['dev_file'],
        tokenizer,
        batch_size=config['batch_size'],
        eval_batch_size=config['eval_batch_size'],
        max_length=config['max_length'],
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Dev batches: {len(dev_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model(config).to(device)
    
    # Setup optimizer and scheduler
    num_training_steps = len(train_loader) * config['num_epochs'] // config['gradient_accumulation_steps']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    scaler = GradScaler(enabled=config['mixed_precision'])
    
    print(f"\nTraining Configuration:")
    print(f"  Total training steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size (effective): {config['batch_size'] * config['gradient_accumulation_steps']}")
    
    # Training loop
    best_spearman = -1.0
    best_checkpoint_path = None
    global_step = 0
    training_history = []
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scaler, config, device, epoch + 1, global_step
        )
        scheduler.step()
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        
        # Validate
        print("\nValidating...")
        val_metrics = validate(model, dev_loader, device, config)
        
        print("\nValidation Metrics:")
        print(format_metrics(val_metrics))
        
        # Save best checkpoint
        if val_metrics['spearman'] > best_spearman:
            best_spearman = val_metrics['spearman']
            best_checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f"best_model.pt"
            )
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"\n✓ New best model saved (Spearman: {best_spearman:.4f})")
        
        # Record history
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
        }
        training_history.append(epoch_history)
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(
            config['checkpoint_dir'],
            f"epoch_{epoch + 1}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Best Spearman correlation: {best_spearman:.4f}")
    print(f"Best model saved to: {best_checkpoint_path}")
    
    # Save training history
    history_path = os.path.join(config['log_dir'], "training_history.json")
    with open(history_path, 'w') as f:
        # Convert tensors to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            return obj
        
        json.dump(convert_to_serializable(training_history), f, indent=2)
    
    print(f"Training history saved to: {history_path}")
    
    return best_checkpoint_path


if __name__ == "__main__":
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    
    # Run training
    best_model_path = train(config)
