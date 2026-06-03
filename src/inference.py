"""
Inference script for homonym sense plausibility prediction.
Loads trained model and predicts plausibility scores for new samples.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, List, Optional
from transformers import AutoTokenizer

# Import custom modules
from model import HomonymRegressionModel
from metrics import denormalize_scores


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are proper types
    config['max_length'] = int(config.get('max_length', 512))
    
    return config


class InferenceModel:
    """Wrapper for inference on new samples."""
    
    def __init__(self, checkpoint_path: str, config: Dict):
        """
        Initialize inference model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration dictionary
        """
        self.config = config
        self.device = config['device']
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = HomonymRegressionModel(model_name=config['model_name']).to(self.device)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"Model loaded from: {checkpoint_path}")
    
    def predict_single(
        self,
        precontext: str,
        sentence: str,
        meaning: str,
        ending: Optional[str] = None,
    ) -> Dict:
        """
        Predict plausibility score for a single sample.
        
        Args:
            precontext: Background context
            sentence: Target sentence containing the homonym
            meaning: Textual definition of the target meaning
            ending: Optional resolution text
        
        Returns:
            Dictionary with predicted score and confidence
        """
        # Construct input
        context = f"{precontext} {sentence}"
        if ending:
            context += f" {ending}"
        
        input_text = f"{context} [SEP] {meaning}"
        
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            max_length=self.config['max_length'],
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        token_type_ids = encoded.get('token_type_ids', torch.zeros_like(input_ids)).to(self.device)
        
        # Predict
        with torch.no_grad():
            with autocast(enabled=self.config['mixed_precision']):
                logit = self.model(input_ids, attention_mask, token_type_ids)
        
        # Denormalize
        logit_normalized = logit.cpu().item()
        logit_denorm = denormalize_scores(
            torch.tensor([[logit_normalized]]),
            target_min=self.config['target_min'],
            target_max=self.config['target_max'],
        ).item()
        
        return {
            'predicted_score': logit_denorm,
            'normalized_score': logit_normalized,
        }
    
    def predict_batch(
        self,
        samples: List[Dict],
    ) -> List[Dict]:
        """
        Predict plausibility scores for multiple samples.
        
        Args:
            samples: List of dicts with keys: precontext, sentence, meaning, [ending]
        
        Returns:
            List of dicts with predictions
        """
        results = []
        for sample in samples:
            result = self.predict_single(
                precontext=sample.get('precontext', ''),
                sentence=sample.get('sentence', ''),
                meaning=sample.get('meaning', sample.get('judged_meaning', '')),
                ending=sample.get('ending'),
            )
            result['sample'] = sample
            results.append(result)
        
        return results


def demo_inference(config: Dict):
    """Run demo inference with example samples."""
    print("\n" + "="*60)
    print("Inference Demo")
    print("="*60)
    
    # Load model
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    inference_model = InferenceModel(checkpoint_path, config)
    
    # Example samples (using patterns from the actual data)
    examples = [
        {
            'precontext': 'The laboratory equipment was carefully arranged on the bench.',
            'sentence': 'The researchers increased the potential gradually.',
            'meaning': 'the difference in electrical charge between two points in a circuit expressed in volts',
            'ending': 'They monitored the voltage meter closely.',
        },
        {
            'precontext': 'The old machine hummed in the corner of the workshop.',
            'sentence': 'The potential couldn\'t be measured.',
            'meaning': 'the likelihood or possibility that something will happen',
            'ending': 'She collected a battery reader and looked on earnestly.',
        },
        {
            'precontext': 'The detectives arrived at the abandoned train station early in the morning.',
            'sentence': 'They followed the track.',
            'meaning': 'a pair of parallel rails providing a runway for wheels',
            'ending': 'They began to run along the abandoned railway line.',
        },
    ]
    
    print(f"\nRunning inference on {len(examples)} examples...\n")
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"  Context: {example['precontext']} {example['sentence']}")
        print(f"  Meaning: {example['meaning']}")
        
        result = inference_model.predict_single(
            precontext=example['precontext'],
            sentence=example['sentence'],
            meaning=example['meaning'],
            ending=example.get('ending'),
        )
        
        print(f"  → Predicted Plausibility Score: {result['predicted_score']:.2f}/5.00")
        print()
    
    print("✓ Inference demo complete!")


if __name__ == "__main__":
    config = load_config("config.yaml")
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print(f"Using checkpoint: {checkpoint_path}")
        
        # Interactive inference
        print("\n" + "="*60)
        print("Interactive Inference")
        print("="*60)
        
        inference_model = InferenceModel(checkpoint_path, config)
        
        while True:
            print("\nEnter homonym sense context (or 'quit' to exit):")
            precontext = input("Precontext: ").strip()
            
            if precontext.lower() == 'quit':
                break
            
            sentence = input("Sentence: ").strip()
            meaning = input("Meaning definition: ").strip()
            ending = input("Ending (optional): ").strip()
            
            result = inference_model.predict_single(
                precontext=precontext,
                sentence=sentence,
                meaning=meaning,
                ending=ending if ending else None,
            )
            
            print(f"\n→ Predicted Plausibility Score: {result['predicted_score']:.2f}/5.00")
    else:
        # Run demo
        demo_inference(config)
