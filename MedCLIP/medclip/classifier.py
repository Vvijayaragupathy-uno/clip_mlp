"""
MLP Classification Head for MedCLIP
Inspired by EyeCLIP and similar vision-language models that add trainable MLP heads
for supervised finetuning on downstream classification tasks.
"""

import torch
from torch import nn
from collections import defaultdict


class MLPClassifier(nn.Module):
    """
    MLP-based classifier that sits on top of MedCLIP vision encoder.
    Similar to how EyeCLIP and other CLIP-based models add classification heads.
    
    Architecture:
    Vision Encoder -> [Optional: Projection] -> MLP Head -> Classification Output
    """
    def __init__(
        self,
        vision_model,
        num_classes,
        input_dim=768,
        hidden_dims=[512, 256],
        dropout=0.3,
        mode='multiclass',
        freeze_backbone=False,
        use_projection=False,
        **kwargs
    ):
        """
        Args:
            vision_model: MedCLIP vision encoder (MedCLIPVisionModelViT or MedCLIPVisionModel)
            num_classes: Number of output classes
            input_dim: Dimension of vision encoder output (768 for ViT, 2048 for ResNet50)
            hidden_dims: List of hidden layer dimensions for MLP [512, 256] creates 2 hidden layers
            dropout: Dropout rate between MLP layers
            mode: 'multiclass', 'multilabel', or 'binary'
            freeze_backbone: If True, freeze vision encoder weights during training
            use_projection: If True, use the projection head output (512-dim), else use raw features
        """
        super().__init__()
        self.vision_model = vision_model
        self.num_classes = num_classes
        self.mode = mode.lower()
        self.use_projection = use_projection
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print("Vision encoder frozen - only training MLP head")
        
        # Determine input dimension
        if use_projection:
            mlp_input_dim = 512  # Projection head output
        else:
            mlp_input_dim = input_dim  # Raw encoder output
        
        # Build MLP layers
        layers = []
        prev_dim = mlp_input_dim
        
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
    
    def forward(self, pixel_values, labels=None, return_loss=True, **kwargs):
        """
        Forward pass through vision encoder and MLP head.
        
        Args:
            pixel_values: Input images [batch_size, channels, height, width]
            labels: Ground truth labels (optional)
            return_loss: Whether to compute and return loss
        
        Returns:
            Dictionary with 'logits', 'embedding', and optionally 'loss_value'
        """
        outputs = defaultdict()
        pixel_values = pixel_values.cuda()
        
        # Get embeddings from vision encoder
        if self.use_projection:
            # Use projected embeddings (512-dim)
            img_embeds = self.vision_model(pixel_values, project=True)
        else:
            # Use raw encoder output (768-dim for ViT, 2048 for ResNet)
            img_embeds = self.vision_model(pixel_values, project=False)
        
        # Pass through MLP head
        logits = self.mlp_head(img_embeds)
        
        outputs['embedding'] = img_embeds
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



