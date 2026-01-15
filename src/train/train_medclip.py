"""
 Toxoplasmosis Classification

"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add MedCLIP to path - updated for new folder structure
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, os.path.join(project_root, 'MedCLIP'))

from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip.dataset import SuperviseImageDataset, SuperviseImageCollator
from medclip.trainer import Trainer
from medclip import constants

# Import our MLP classifier
from medclip.classifier import MLPClassifier

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("MedCLIP + MLP for Ocular Toxoplasmosis Classification")
print("=" * 80)
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Training configuration
train_config = {
    'batch_size': 16,  # Smaller batch size for limited data
    'num_epochs': 100,  # More epochs for small dataset
    'warmup': 0.1,
    'lr': 5e-5,  # Lower learning rate for finetuning
    'weight_decay': 0.01,
    'eval_batch_size': 32,
    'eval_steps': 50,  # Evaluate more frequently
    'save_steps': 50,
}

# MLP configuration
mlp_config = {
    'hidden_dims': [512, 256],  # Two hidden layers
    'dropout': 0.4,  # Higher dropout for small dataset
    'freeze_backbone': False,  # Train entire model
    'use_projection': False,  # Use raw 768-dim features
}

# Dataset configuration
dataset_config = {
    'train_data': ['toxoplasmosis-train'],
    'val_data': ['toxoplasmosis-val'],
    'class_names': ['healthy', 'active', 'inactive'],
    'num_classes': 3,
    'mode': 'multiclass',
}

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("Preparing Data")
print("=" * 80)

# Data augmentation for training - aggressive for small dataset
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),  # Add vertical flip for fundus images
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, scale=(0.85, 1.15), translate=(0.1, 0.1)),
    transforms.RandomRotation(15),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])
])

# No augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])
])

# Create datasets
try:
    train_dataset = SuperviseImageDataset(
        datalist=dataset_config['train_data'],
        class_names=dataset_config['class_names'],
        imgtransform=train_transform
    )
    
    val_dataset = SuperviseImageDataset(
        datalist=dataset_config['val_data'],
        class_names=dataset_config['class_names'],
        imgtransform=val_transform
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    print("\n⚠ Make sure you've run 'prepare_toxo_dataset.py' first!")
    sys.exit(1)

# Create data loaders
train_collate_fn = SuperviseImageCollator(mode=dataset_config['mode'])
val_collate_fn = SuperviseImageCollator(mode=dataset_config['mode'])

train_loader = DataLoader(
    train_dataset,
    batch_size=train_config['batch_size'],
    collate_fn=train_collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=train_config['eval_batch_size'],
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

# ============================================================================
# MODEL SETUP
# ============================================================================

print("\n" + "=" * 80)
print("Building Model")
print("=" * 80)

# Load pretrained vision encoder
vision_encoder = MedCLIPVisionModelViT()

# Try to load pretrained MedCLIP weights
try:
    medclip_full = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    medclip_full.from_pretrained()
    vision_encoder = medclip_full.vision_model
    print("✓ Loaded pretrained MedCLIP vision encoder")
except Exception as e:
    print(f"⚠ Could not load pretrained weights: {e}")
    print("  Using randomly initialized vision encoder")

# Create MLP classifier
model = MLPClassifier(
    vision_model=vision_encoder,
    num_classes=dataset_config['num_classes'],
    input_dim=768,  # ViT output dimension
    hidden_dims=mlp_config['hidden_dims'],
    dropout=mlp_config['dropout'],
    mode=dataset_config['mode'],
    freeze_backbone=mlp_config['freeze_backbone'],
    use_projection=mlp_config['use_projection']
)

model = model.cuda()

print(f"\n✓ Model Configuration:")
print(f"  - Backbone: MedCLIPVisionModelViT")
print(f"  - Backbone frozen: {mlp_config['freeze_backbone']}")
print(f"  - MLP architecture: 768 -> {' -> '.join(map(str, mlp_config['hidden_dims']))} -> {dataset_config['num_classes']}")
print(f"  - Dropout: {mlp_config['dropout']}")
print(f"  - Classes: {dataset_config['class_names']}")
print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("Starting Training")
print("=" * 80)

# Wrap model for trainer
class LossWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values, labels, **kwargs):
        return self.model(pixel_values=pixel_values, labels=labels, return_loss=True)

loss_model = LossWrapper(model)
loss_model.cuda()

# Setup training objectives
train_objectives = [
    (train_loader, loss_model, 1),
]

# Model save path
model_save_path = './checkpoints/toxoplasmosis_medclip_mlp'
os.makedirs(model_save_path, exist_ok=True)

# Create trainer and start training
trainer = Trainer()
trainer.train(
    model,
    train_objectives=train_objectives,
    warmup_ratio=train_config['warmup'],
    epochs=train_config['num_epochs'],
    optimizer_params={'lr': train_config['lr']},
    output_path=model_save_path,
    evaluation_steps=train_config['eval_steps'],
    weight_decay=train_config['weight_decay'],
    save_steps=train_config['save_steps'],
    evaluator=None,
    eval_dataloader=val_loader,
    use_amp=True,
)

print("\n" + "=" * 80)
print("Training Complete!")
print(f"Model saved to: {model_save_path}")
print("=" * 80)

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("Evaluating on Validation Set")
print("=" * 80)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch['pixel_values'].cuda()
        labels = batch['labels'].cuda()
        
        outputs = model(pixel_values=pixel_values, labels=labels, return_loss=False)
        logits = outputs['logits']
        
        predictions = torch.argmax(logits, dim=1)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print classification report
print("\nClassification Report:")
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=dataset_config['class_names'],
    digits=4
))

# Print confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)
print(f"\nClass order: {dataset_config['class_names']}")

# Calculate accuracy
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n✓ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "=" * 80)
print("✓ All Done!")
print("=" * 80)
