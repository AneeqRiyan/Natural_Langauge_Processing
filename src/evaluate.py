"""
Evaluation script for homonym sense plausibility prediction model.
Loads best checkpoint and computes final metrics on dev set.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Dict
import pandas as pd

# Import custom modules
from data import get_dataloaders, HomonymDataset
from model import HomonymRegressionModel
from metrics import compute_regression_metrics, format_metrics, denormalize_scores


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are proper types
    config['batch_size'] = int(config.get('batch_size', 16))
    config['learning_rate'] = float(config.get('learning_rate', 2e-5))
    config['weight_decay'] = float(config.get('weight_decay', 1e-2))
    config['max_length'] = int(config.get('max_length', 512))
    config['eval_batch_size'] = int(config.get('eval_batch_size', 32))
    
    return config


def evaluate(checkpoint_path: str, config: Dict):
    """
    Evaluate model on dev set and generate detailed report.
    """
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    
    # Setup device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    _, dev_loader = get_dataloaders(
        config['train_file'],
        config['dev_file'],
        tokenizer,
        batch_size=config['batch_size'],
        eval_batch_size=config['eval_batch_size'],
        max_length=config['max_length'],
    )
    
    # Load model
    print("Loading model...")
    model = HomonymRegressionModel(model_name=config['model_name']).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    
    # Evaluate
    print("\nEvaluating on dev set...")
    
    all_predictions = []
    all_targets = []
    all_stdevs = []
    all_sample_ids = []
    all_averages = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating", leave=True):
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
            all_sample_ids.extend(batch['sample_id'])
            all_averages.append(batch['average'].cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    stdevs = torch.cat(all_stdevs)
    averages = torch.cat(all_averages)
    
    # Compute metrics
    avg_loss = total_loss / len(dev_loader)
    metrics = compute_regression_metrics(predictions, targets, stdevs)
    metrics['loss'] = avg_loss
    
    print(f"\nFinal Metrics:")
    print(format_metrics(metrics))
    
    # Generate detailed report
    predictions_denorm = denormalize_scores(predictions)
    
    report_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'predicted_score': predictions_denorm.numpy(),
        'human_average': averages.numpy(),
        'human_stdev': stdevs.numpy(),
        'absolute_error': torch.abs(predictions_denorm - averages).numpy(),
        'within_stdev': (torch.abs(predictions_denorm - averages) <= stdevs).numpy(),
    })
    
    # Sort by error
    report_df = report_df.sort_values('absolute_error', ascending=False)
    
    # Save report
    report_path = os.path.join(config['log_dir'], "evaluation_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Print summary statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {predictions_denorm.mean():.2f}")
    print(f"  Std: {predictions_denorm.std():.2f}")
    print(f"  Min: {predictions_denorm.min():.2f}")
    print(f"  Max: {predictions_denorm.max():.2f}")
    
    print(f"\nHuman Average Statistics:")
    print(f"  Mean: {averages.mean():.2f}")
    print(f"  Std: {averages.std():.2f}")
    print(f"  Min: {averages.min():.2f}")
    print(f"  Max: {averages.max():.2f}")
    
    # Top 10 best predictions
    print(f"\nTop 10 Best Predictions (lowest error):")
    print(report_df.head(10).to_string())
    
    # Top 10 worst predictions
    print(f"\nTop 10 Worst Predictions (highest error):")
    print(report_df.tail(10).to_string())
    
    return metrics, report_df


if __name__ == "__main__":
    config_path = "config.yaml"
    checkpoint_path = "checkpoints/best_model.pt"
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    config = load_config(config_path)
    metrics, report = evaluate(checkpoint_path, config)
