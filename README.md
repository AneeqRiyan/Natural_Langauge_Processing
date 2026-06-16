# Homonym Sense Plausibility Prediction with DeBERTa

A complete PyTorch + Hugging Face pipeline for fine-tuning **DeBERTa-v3-large** to predict plausibility scores (1-5) for homonym meanings in textual context.

## Overview

**Task**: Given a story context and a homonym meaning definition, predict how plausible that meaning is in the given context.

**Approach**: 
- Fine-tune a pretrained DeBERTa-v3-large model as a regression model
- Input: concatenation of context (precontext + sentence + optional ending) + meaning definition
- Output: continuous plausibility score [1, 5]
- Training: MSE loss with mixed precision and gradient accumulation
- Evaluation: Spearman correlation + AWSD (Accuracy Within Standard Deviation)

## Project Structure

```
code/
├── config.yaml                  # Hyperparameters and settings
├── requirements.txt             # Python dependencies
├── validate.py                  # Environment and data validation
├── predict.py                   # Required submission entry point for predictions (SemEval Task 5 interface)
├── train.json                   # Training data (~200+ samples)
├── dev.json                     # Development/validation data (~200+ samples)
├── src/
│   ├── __init__.py
│   ├── data.py                  # DataLoader and data utilities
│   ├── model.py                 # DeBERTa regression model
│   ├── metrics.py               # Evaluation metrics
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference interface
├── checkpoints/                 # Model checkpoints (created during training)
└── logs/                        # Training logs and evaluation reports
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate Setup

Run validation to check environment, data, and model:

```bash
python validate.py
```

You should see:
- ✓ All config files found
- ✓ Data files loaded correctly
- ✓ Model initialized successfully
- ✓ DataLoaders working

### 3. Train the Model

```bash
python src/train.py config.yaml
```

The script will:
- Load training and dev data
- Initialize DeBERTa-v3-large with frozen base layers
- Train for 3 epochs with mixed precision and gradient accumulation
- Save best checkpoint (by Spearman correlation) to `checkpoints/best_model.pt`
- Log training history to `logs/training_history.json`

**Expected output**:
```
Training Configuration:
  Total training steps: 380
  Warmup steps: 38
  Learning rate: 2e-05
  Batch size (effective): 64

Epoch 1/3
  Train Loss: 0.1234
  Spearman Correlation: 0.4567
  AWSD: 65.43%

...

Training Complete!
Total time: X.XX hours
Best Spearman correlation: 0.5234
Best model saved to: checkpoints/best_model.pt
```

### 4. Evaluate on Dev Set

```bash
python src/evaluate.py checkpoints/best_model.pt
```

Generates:
- Console report with Spearman, AWSD, and other metrics
- CSV file with per-sample predictions vs. human averages
- Best and worst predictions

**Example output**:
```
Final Metrics:
  MSE: 0.0456
  RMSE: 0.2135
  MAE: 0.3421
  Spearman Correlation: 0.5234 (p=1.23e-45)
  AWSD: 65.43% (131/200)
```

### 5. Make Predictions on New Samples

#### A. Generate Submission Predictions (Required Format)

To generate predictions for evaluation (e.g., for SemEval submission) in the correct format, run the required entry-point script `predict.py` at the root of the project:

```bash
python predict.py <input_json> <output_jsonl>
```

For example, to predict on `dev.json` and save to `predictions.jsonl`:

```bash
python predict.py dev.json predictions.jsonl
```

This script will:
- Load the configuration from `config.yaml`
- Load the best model checkpoint from `checkpoints/best_model.pt`
- Generate integer predictions rounded to the range `[1, 5]`
- Output predictions in JSONL format, containing one `{"id": "...", "prediction": N}` object per line.

#### B. Batch Inference Demo

To run a demonstration on sample sentences:

```bash
python src/inference.py checkpoints/best_model.pt
```

This runs a demo with 3 example samples showing predicted continuous plausibility scores.

#### C. Interactive Inference

To run inference programmatically in your own script:

```python
import yaml
from pathlib import Path
from src.inference import InferenceModel

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize inference model
inference_model = InferenceModel("checkpoints/best_model.pt", config)

# Predict score for a single sample
result = inference_model.predict_single(
    precontext="The laboratory equipment was carefully arranged.",
    sentence="The researchers increased the potential gradually.",
    meaning="the difference in electrical charge between two points",
    ending="They monitored the voltage meter closely."
)

print(f"Predicted score: {result['predicted_score']:.2f}/5.00")
```

## Configuration

Edit `config.yaml` to adjust:

**Model**:
- `model_name`: DeBERTa variant (default: `microsoft/deberta-v3-large`)
- `max_length`: Maximum sequence length (default: 512)
- `hidden_dropout_prob`: Dropout in regression head (default: 0.2)

**Training**:
- `batch_size`: Batch size per accumulation step (default: 16 for CPU, can increase to 32-64 for GPU)
- `gradient_accumulation_steps`: How many batches before optimizer step (default: 4, effective batch = 16 * 4 = 64)
- `num_epochs`: Number of training epochs (default: 3)
- `learning_rate`: Initial learning rate (default: 2e-5)
- `weight_decay`: L2 regularization (default: 1e-2)
- `warmup_ratio`: Fraction of steps for linear warmup (default: 0.1)

**Hardware**:
- `device`: `"cpu"` or `"cuda"` (default: `"cpu"`, auto-falls back if GPU unavailable)
- `mixed_precision`: Enable mixed precision training (default: `true`, reduces memory and speeds up training)

**Data**:
- `train_file`, `dev_file`: Paths to training/dev JSON files
- `target_min`, `target_max`: Score range (default: 1.0 to 5.0)

## Architecture

### Model

```
Input: [precontext] [sentence] [ending] [SEP] [judged_meaning]
  ↓
DeBERTa-v3-large (frozen base layers)
  ↓
[CLS] token → LayerNorm → Dropout(0.2) → Linear(768→1)
  ↓
Output: normalized score ∈ [0, 1] → denormalized to [1, 5]
```

**Why frozen layers?**
- Reduces trainable parameters from 300M+ to ~1-2M
- Faster training, lower memory
- Reduces overfitting on small dataset (~200 samples)
- Leverages strong pretrained representations

### Training

- **Loss**: MSE on normalized scores [0, 1]
- **Optimizer**: AdamW with weight decay (1e-2)
- **Scheduler**: Linear warmup (10% of steps) → decay to 0
- **Mixed Precision**: `torch.cuda.amp.autocast()` for memory efficiency
- **Gradient Accumulation**: Simulates batch size 64 on CPU with batch_size=16
- **Validation**: Every ~500 steps, saves best checkpoint by Spearman correlation

### Metrics

**Spearman Rank Correlation**: Measures rank-order consistency between predictions and human scores.
- Range: -1 to 1 (higher is better)
- Target: ≥ 0.5 for this task

**Accuracy Within Standard Deviation (AWSD)**: Percentage of predictions within ± one human stdev of the average.
- Accounts for inter-annotator disagreement
- Range: 0% to 100% (higher is better)
- Target: ≥ 60% for this task

## Performance Expectations

On similar datasets:
- **Spearman Correlation**: 0.45 - 0.55 (reasonable for this task)
- **AWSD**: 55% - 70%
- **MAE**: 0.30 - 0.50 on [1, 5] scale

If performance is lower:
1. Check data quality and ensure JSON files are valid
2. Increase `num_epochs` to 4-5
3. Reduce `learning_rate` to 1e-5
4. Try unfreezing more layers (reduce `num_frozen_layers` in `model.py`)

If overfitting (val loss increases after early epochs):
1. Increase `gradient_accumulation_steps` for effective larger batch
2. Increase `weight_decay`
3. Use more aggressive early stopping

## GPU Transition

When you have GPU access, only change in `config.yaml`:
```yaml
device: "cuda"
batch_size: 32  # or higher depending on GPU memory
```

No code changes needed! Mixed precision is enabled by default.

## File Formats

### train.json / dev.json Structure

```json
{
  "0": {
    "homonym": "potential",
    "judged_meaning": "the difference in electrical charge between two points in a circuit expressed in volts",
    "precontext": "The old machine hummed in the corner of the workshop...",
    "sentence": "The potential couldn't be measured.",
    "ending": "She collected a battery reader and looked on earnestly...",
    "choices": [4, 5, 2, 3, 1],
    "average": 3.0,
    "stdev": 1.5811388300841898,
    "sample_id": "1843",
    "example_sentence": "The circuit has a high potential difference."
  },
  ...
}
```

**Key fields**:
- `precontext`: Background narrative
- `sentence`: Target sentence with homonym
- `ending`: Optional resolution text
- `judged_meaning`: Definition of target meaning
- `average`: Mean human rating (target for regression)
- `stdev`: Annotator agreement standard deviation
- `sample_id`: Unique identifier

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` (e.g., 8 or 4)
- Reduce `max_length` (e.g., 256)
- Disable mixed precision: `mixed_precision: false`
- Use CPU (much slower but always works)

### Poor Performance
- Check data quality: `python validate.py`
- Increase training epochs: `num_epochs: 5`
- Train longer: set `eval_steps: 200` to validate more frequently
- Check learning rate: might need 1e-5 or 5e-5 depending on dataset

### Training Unstable (loss increasing, NaNs)
- Reduce learning rate: `learning_rate: 1e-5`
- Increase weight decay: `weight_decay: 0.1`
- Disable mixed precision: `mixed_precision: false`
- Check for data issues: validate tokenization

### Can't Find Checkpoint
- Ensure you've completed at least 1 epoch of training
- Check that `checkpoints/` directory exists and contains `.pt` files
- Default best model: `checkpoints/best_model.pt`

## References

- **DeBERTa**: https://huggingface.co/microsoft/deberta-v3-large
- **Hugging Face Transformers**: https://huggingface.co/transformers/
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html

## License

This project is provided as-is for educational and research purposes.

---

**Questions?** Check the code comments in each module for detailed explanations and examples.
