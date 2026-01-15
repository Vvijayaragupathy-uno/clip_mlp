"""
CLIP Classifier with MLP Head
Same architecture as MedCLIP classifier but for OpenAI CLIP
"""

import torch
from torch import nn
from collections import defaultdict


class CLIPClassifier(nn.Module):
    """
    MLP-based classifier for OpenAI CLIP vision encoder.
    Same architecture as our MedCLIP classifier.
    
    Architecture:
    CLIP Vision Encoder -> MLP Head -> Classification Output
    """
    def __init__(
        self,
        clip_model,
        num_classes,
        input_dim=512,  # CLIP ViT-B/32 output is 512-dim
        hidden_dims=[512, 256],
        dropout=0.3,
        mode='multiclass',
        freeze_backbone=False,
        **kwargs
    ):
        """
        Args:
            clip_model: OpenAI CLIP model
            num_classes: Number of output classes
            input_dim: Dimension of CLIP vision encoder output (512 for ViT-B/32, 768 for ViT-B/16)
            hidden_dims: List of hidden layer dimensions for MLP
            dropout: Dropout rate between MLP layers
            mode: 'multiclass', 'multilabel', or 'binary'
            freeze_backbone: If True, freeze CLIP encoder weights during training
        """
        super().__init__()
        self.clip_model = clip_model
        self.num_classes = num_classes
        self.mode = mode.lower()
        
        # Freeze CLIP backbone if specified
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("CLIP encoder frozen - only training MLP head")
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        if num_classes > 2:
            layers.append(nn.Linear(prev_dim, num_classes))
        else:
            layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp_head = nn.Sequential(*layers)
        
        # Set loss function
        assert self.mode in ['multiclass', 'multilabel', 'binary']
        if num_classes > 2:
            if self.mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:  # multilabel
                self.loss_fn = nn.BCEWithLogitsLoss()
        else:  # binary
            self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, images, labels=None, return_loss=True, **kwargs):
        """
        Forward pass through CLIP encoder and MLP head.
        
        Args:
            images: Input images [batch_size, 3, 224, 224]
            labels: Ground truth labels (optional)
            return_loss: Whether to compute and return loss
        
        Returns:
            Dictionary with 'logits', 'embedding', and optionally 'loss_value'
        """
        outputs = defaultdict()
        
        # Get embeddings from CLIP vision encoder
        with torch.no_grad() if not self.training else torch.enable_grad():
            image_features = self.clip_model.encode_image(images)
            # Normalize features (CLIP standard)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Pass through MLP head
        logits = self.mlp_head(image_features.float())
        
        outputs['embedding'] = image_features
        outputs['logits'] = logits
        
        # Compute loss if labels provided
        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1:
                labels = labels.view(-1, 1)
            if self.mode == 'multiclass':
                labels = labels.flatten().long()
            loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        
        return outputs
