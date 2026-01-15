"""
Train OpenAI CLIP with MLP Classification Head for Toxoplasmosis Classification

"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import pandas as pd
import clip  # pip install git+https://github.com/openai/CLIP.git

# Import our CLIP classifier
from clip_classifier import CLIPClassifier

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("OpenAI CLIP + MLP for Ocular Toxoplasmosis Classification")
print("=" * 80)
print(f"Device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

train_config = {
    'batch_size': 16,
    'num_epochs': 100,
    'lr': 1e-3,  # Higher LR for MLP-only training
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,  # Gradient clipping
}

mlp_config = {
    'hidden_dims': [512, 256],
    'dropout': 0.4,
    'freeze_backbone': True,  # Freeze CLIP, train only MLP
}

dataset_config = {
    'class_names': ['healthy', 'active', 'inactive'],
    'num_classes': 3,
    'mode': 'multiclass',
}

# ============================================================================
# DATASET
# ============================================================================

class ToxoDataset(Dataset):
    """Custom dataset for toxoplasmosis images"""
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        self.transform = transform
        self.class_to_idx = {'healthy': 0, 'active': 1, 'inactive': 2}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['imgpath']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = np.argmax([row['healthy'], row['active'], row['inactive']])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# LOAD CLIP MODEL
# ============================================================================

print("\n" + "=" * 80)
print("Loading CLIP Model")
print("=" * 80)

# Load CLIP model and preprocessing
clip_model, preprocess = clip.load("ViT-B/32", device=device)
print("✓ Loaded CLIP ViT-B/32")

# Get CLIP's image preprocessing
# We'll use CLIP's standard preprocessing for validation
# But add augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, scale=(0.85, 1.15), translate=(0.1, 0.1)),
    transforms.RandomRotation(15),
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP's normalization
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n" + "=" * 80)
print("Preparing Data")
print("=" * 80)

# Note: Using processed data from new folder structure
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
data_dir = os.path.join(project_root, 'data', 'processed')

train_dataset = ToxoDataset(
    os.path.join(data_dir, 'toxoplasmosis-train-meta.csv'),
    transform=train_transform
)

val_dataset = ToxoDataset(
    os.path.join(data_dir, 'toxoplasmosis-val-meta.csv'),
    transform=val_transform
)

print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Validation samples: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=train_config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ============================================================================
# BUILD MODEL
# ============================================================================

print("\n" + "=" * 80)
print("Building Model")
print("=" * 80)

# Create classifier with MLP head
model = CLIPClassifier(
    clip_model=clip_model,
    num_classes=dataset_config['num_classes'],
    input_dim=512,  # CLIP ViT-B/32 outputs 512-dim features
    hidden_dims=mlp_config['hidden_dims'],
    dropout=mlp_config['dropout'],
    mode=dataset_config['mode'],
    freeze_backbone=mlp_config['freeze_backbone']
)

model = model.to(device)

print(f"\n✓ Model Configuration:")
print(f"  - Backbone: CLIP ViT-B/32")
print(f"  - Backbone frozen: {mlp_config['freeze_backbone']}")
print(f"  - MLP architecture: 512 -> {' -> '.join(map(str, mlp_config['hidden_dims']))} -> {dataset_config['num_classes']}")
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

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config['lr'],
    weight_decay=train_config['weight_decay']
)

# Scheduler
total_steps = len(train_loader) * train_config['num_epochs']
warmup_steps = int(0.1 * total_steps)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps
)

# Training loop
best_acc = 0.0
save_dir = './checkpoints/toxoplasmosis_clip_mlp'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(train_config['num_epochs']):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images, labels=labels, return_loss=True)
        loss = outputs['loss_value']
        
        loss.backward()
        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{train_config['num_epochs']}] "
                  f"Step [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    if (epoch + 1) % 5 == 0:
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images, return_loss=False)
                predictions = torch.argmax(outputs['logits'], dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1} - Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*80}\n")
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
            print(f"✓ Saved best model with accuracy: {best_acc:.4f}")

print("\n" + "=" * 80)
print("Training Complete!")
print(f"Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
print("=" * 80)

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("Final Evaluation on Validation Set")
print("=" * 80)

# Load best model
model.load_state_dict(torch.load(f'{save_dir}/best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images, return_loss=False)
        predictions = torch.argmax(outputs['logits'], dim=1)
        
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
print(f"✓ Best model saved to: {save_dir}/best_model.pth")
print("=" * 80)
