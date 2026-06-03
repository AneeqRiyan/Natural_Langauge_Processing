"""
Evaluation metrics for homonym sense plausibility prediction.
Implements Spearman correlation and Accuracy Within Standard Deviation (AWSD).
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Tuple


def denormalize_scores(scores: torch.Tensor, target_min: float = 1.0, target_max: float = 5.0) -> torch.Tensor:
    """
    Denormalize scores from [0, 1] to [target_min, target_max].
    
    Args:
        scores: Normalized scores [0, 1]
        target_min: Minimum target value
        target_max: Maximum target value
    
    Returns:
        Denormalized scores
    """
    return scores * (target_max - target_min) + target_min


def compute_spearman_correlation(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Spearman rank-order correlation coefficient.
    
    Args:
        predictions: Predicted scores (normalized [0, 1] or unnormalized [1, 5])
        targets: Target scores (same scale as predictions)
    
    Returns:
        Dict with 'spearman' and 'pvalue' keys
    """
    # Convert to numpy
    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Compute Spearman correlation
    correlation, pvalue = spearmanr(targets_np, preds_np)
    
    # Handle NaN case (e.g., if all values are identical)
    if np.isnan(correlation):
        correlation = 0.0
        pvalue = 1.0
    
    return {
        'spearman': float(correlation),
        'pvalue': float(pvalue),
    }


def compute_awsd(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    stdevs: torch.Tensor,
    epsilon: float = 0.1,
) -> Dict[str, float]:
    """
    Compute Accuracy Within Standard Deviation (AWSD).
    
    A prediction is considered accurate if |prediction - target| <= target_stdev.
    This accounts for inter-annotator disagreement.
    
    Args:
        predictions: Predicted scores (denormalized, same scale as targets)
        targets: Target scores (averages from human annotations)
        stdevs: Standard deviations from human annotations
        epsilon: Minimum stdev to avoid division by zero
    
    Returns:
        Dict with 'awsd' (float 0-1), 'awsd_percentage' (float 0-100), and 'num_within' (int)
    """
    # Ensure stdevs have minimum value
    stdevs = torch.clamp(stdevs, min=epsilon)
    
    # Compute absolute differences
    differences = torch.abs(predictions - targets)
    
    # Check if within one standard deviation
    within_stdev = differences <= stdevs
    
    # Calculate percentage
    awsd_percentage = 100 * within_stdev.float().mean().item()
    awsd_ratio = within_stdev.float().mean().item()
    num_within = within_stdev.sum().item()
    
    return {
        'awsd': awsd_ratio,
        'awsd_percentage': awsd_percentage,
        'num_within': int(num_within),
        'total': len(predictions),
    }


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    stdevs: torch.Tensor,
    target_min: float = 1.0,
    target_max: float = 5.0,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Assumes predictions are normalized [0, 1] and targets are also normalized.
    Denormalizes both to original scale [target_min, target_max] for metrics calculation.
    
    Args:
        predictions: Predicted scores (normalized [0, 1])
        targets: Target average scores (normalized [0, 1])
        stdevs: Standard deviations (denormalized, original scale)
        target_min: Minimum target value (1.0 for 1-5 scale)
        target_max: Maximum target value (5.0 for 1-5 scale)
    
    Returns:
        Dictionary with all metrics
    """
    # Denormalize predictions and targets back to original scale
    predictions_denorm = denormalize_scores(predictions, target_min, target_max)
    targets_denorm = denormalize_scores(targets, target_min, target_max)
    
    # Compute MSE (on normalized scale)
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = np.sqrt(mse)
    
    # Compute MAE (on denormalized scale for interpretability)
    mae = torch.mean(torch.abs(predictions_denorm - targets_denorm)).item()
    
    # Compute Spearman correlation
    spearman_results = compute_spearman_correlation(predictions_denorm, targets_denorm)
    
    # Compute AWSD
    awsd_results = compute_awsd(predictions_denorm, targets_denorm, stdevs)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'spearman': spearman_results['spearman'],
        'spearman_pvalue': spearman_results['pvalue'],
        'awsd': awsd_results['awsd'],
        'awsd_percentage': awsd_results['awsd_percentage'],
        'num_within_stdev': awsd_results['num_within'],
        'total_samples': awsd_results['total'],
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as readable string."""
    lines = [
        f"  MSE: {metrics['mse']:.4f}",
        f"  RMSE: {metrics['rmse']:.4f}",
        f"  MAE: {metrics['mae']:.4f}",
        f"  Spearman Correlation: {metrics['spearman']:.4f} (p={metrics['spearman_pvalue']:.4e})",
        f"  AWSD: {metrics['awsd_percentage']:.2f}% ({metrics['num_within_stdev']}/{metrics['total_samples']})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    """Test metrics computation."""
    # Create synthetic predictions and targets
    batch_size = 100
    
    # Normalized predictions and targets [0, 1]
    predictions = torch.rand(batch_size)  # Random predictions
    targets = torch.rand(batch_size)  # Random targets
    stdevs = torch.rand(batch_size) * 0.5 + 0.5  # Stdevs between 0.5 and 1.0 (original scale)
    
    # Compute metrics
    metrics = compute_regression_metrics(predictions, targets, stdevs)
    
    print("Test Metrics:")
    print(format_metrics(metrics))
    
    # Test perfect prediction
    print("\n\nTest Perfect Prediction:")
    perfect_predictions = targets.clone()
    perfect_metrics = compute_regression_metrics(perfect_predictions, targets, stdevs)
    print(format_metrics(perfect_metrics))
