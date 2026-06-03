"""
Homonym Sense Plausibility Prediction - DeBERTa Fine-tuning Package

This package implements a complete pipeline for training and evaluating a DeBERTa-v3-large
model to predict plausibility scores (1-5) for homonym meanings in context.

Modules:
- data.py: Data loading, preprocessing, and DataLoader utilities
- model.py: DeBERTa regression model architecture
- metrics.py: Evaluation metrics (Spearman, AWSD)
- train.py: Training loop with mixed precision and gradient accumulation
- evaluate.py: Evaluation script with detailed reporting
- inference.py: Inference interface for new predictions
"""

__version__ = "1.0.0"
