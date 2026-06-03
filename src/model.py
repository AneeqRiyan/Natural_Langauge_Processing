"""
DeBERTa regression model for homonym sense plausibility prediction.
Fine-tunes DeBERTa-v3-large with a custom regression head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict


class HomonymRegressionModel(nn.Module):
    """
    DeBERTa-v3-large fine-tuned as a regression model for plausibility prediction.
    
    Architecture:
    - Pretrained DeBERTa-v3-large (frozen except final layers)
    - Regression head: LayerNorm + Dropout + Linear(768 -> 1)
    - Output: single float value (normalized to [0, 1], denormalized to [1, 5] at inference)
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        dropout_prob: float = 0.2,
        num_frozen_layers: int = 19,  # Freeze all but last layer of DeBERTa base
    ):
        """
        Args:
            model_name: Hugging Face model identifier
            dropout_prob: Dropout probability in regression head
            num_frozen_layers: Number of DeBERTa layers to freeze from the beginning
        """
        super().__init__()
        
        # Load pretrained model
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size  # Get actual hidden size from config
        
        # Freeze base layers for efficient fine-tuning
        self._freeze_layers(num_frozen_layers)
        
        # Regression head
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.regression_head = nn.Linear(self.hidden_size, 1)
        
        # Initialize regression head weights
        nn.init.normal_(self.regression_head.weight, std=0.02)
        nn.init.zeros_(self.regression_head.bias)
    
    def _freeze_layers(self, num_frozen_layers: int):
        """Freeze the first N layers of DeBERTa to reduce parameters."""
        total_layers = len(self.encoder.encoder.layer)
        
        if num_frozen_layers > 0:
            # Freeze embeddings and initial layers
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            
            # Freeze specified number of transformer layers
            for i in range(min(num_frozen_layers, total_layers)):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        print(f"Frozen {num_frozen_layers} layers out of {total_layers} transformer layers")
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through DeBERTa and regression head.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length]
        
        Returns:
            logits: Regression outputs [batch_size, 1]
        """
        # Pass through DeBERTa encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Apply regression head
        x = self.layer_norm(cls_output)
        x = self.dropout(x)
        logits = self.regression_head(x)  # [batch_size, 1]
        
        return logits.squeeze(-1)  # Return shape [batch_size]


def create_model(config: Dict) -> HomonymRegressionModel:
    """
    Factory function to create model from config dict.
    
    Args:
        config: Configuration dictionary with model_name, hidden_dropout_prob
    
    Returns:
        Initialized HomonymRegressionModel
    """
    model = HomonymRegressionModel(
        model_name=config.get('model_name', 'microsoft/deberta-v3-large'),
        dropout_prob=config.get('hidden_dropout_prob', 0.2),
        num_frozen_layers=19,
    )
    
    trainable_params = model.get_trainable_params()
    total_params = model.get_total_params()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


if __name__ == "__main__":
    """Quick sanity check."""
    import torch
    
    print("Creating model...")
    model = HomonymRegressionModel()
    
    # Test forward pass
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, 28996, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print("\n✓ Model forward pass successful!")
