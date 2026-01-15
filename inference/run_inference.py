"""
Simple inference script for both MedCLIP and OpenAI CLIP models
Runs inference on sample images and saves results
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../MedCLIP'))

from clip_classifier import CLIPClassifier
from medclip.modeling_medclip import MedCLIPVisionModelViT
from medclip.classifier import MLPClassifier
from medclip import constants
import clip

# Configuration
CLASS_NAMES = ['healthy', 'active', 'inactive']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("CLIP Models Inference")
print("=" * 80)
print(f"Device: {DEVICE}\n")

# Image preprocessing
medclip_transform = transforms.Compose([
    transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])
])

clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

# Load MedCLIP model
print("Loading MedCLIP model...")
medclip_vision = MedCLIPVisionModelViT()
medclip_model = MLPClassifier(
    vision_model=medclip_vision,
    num_classes=3,
    input_dim=768,
    hidden_dims=[512, 256],
    dropout=0.4,
    mode='multiclass',
    freeze_backbone=False,
    use_projection=False
)
medclip_checkpoint = torch.load('models/medclip_best.bin', map_location=DEVICE)
medclip_model.load_state_dict(medclip_checkpoint)
medclip_model = medclip_model.to(DEVICE)
medclip_model.eval()
print("✓ MedCLIP loaded\n")

# Load OpenAI CLIP model
print("Loading OpenAI CLIP model...")
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
openai_clip = CLIPClassifier(
    clip_model=clip_model,
    num_classes=3,
    input_dim=512,
    hidden_dims=[512, 256],
    dropout=0.4,
    mode='multiclass',
    freeze_backbone=True
)
clip_checkpoint = torch.load('models/clip_best.pth', map_location=DEVICE)
openai_clip.load_state_dict(clip_checkpoint)
openai_clip = openai_clip.to(DEVICE)
openai_clip.eval()
print("✓ OpenAI CLIP loaded\n")

# Run inference on sample images
sample_dir = 'sample_images'
results = []

print("=" * 80)
print("Running Inference on Sample Images")
print("=" * 80)

for img_file in sorted(os.listdir(sample_dir)):
    if not img_file.endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    img_path = os.path.join(sample_dir, img_file)
    print(f"\n📷 Processing: {img_file}")
    
    # Load image
    image = Image.open(img_path).convert('RGB')
    
    # MedCLIP inference
    with torch.no_grad():
        medclip_input = medclip_transform(image).unsqueeze(0).to(DEVICE)
        # MedCLIP expects grayscale, convert RGB to grayscale-like format
        if medclip_input.shape[1] == 3:
            medclip_input = medclip_input.mean(dim=1, keepdim=True)
        medclip_input = medclip_input.repeat(1, 3, 1, 1)  # Convert back to 3 channels
        
        medclip_output = medclip_model(pixel_values=medclip_input, return_loss=False)
        medclip_probs = torch.softmax(medclip_output['logits'], dim=1)[0]
        medclip_pred = torch.argmax(medclip_probs).item()
        medclip_conf = medclip_probs[medclip_pred].item()
    
    # OpenAI CLIP inference
    with torch.no_grad():
        clip_input = clip_transform(image).unsqueeze(0).to(DEVICE)
        clip_output = openai_clip(clip_input, return_loss=False)
        clip_probs = torch.softmax(clip_output['logits'], dim=1)[0]
        clip_pred = torch.argmax(clip_probs).item()
        clip_conf = clip_probs[clip_pred].item()
    
    # Print results
    print(f"  MedCLIP:     {CLASS_NAMES[medclip_pred]:10s} ({medclip_conf*100:.1f}% confidence)")
    print(f"  OpenAI CLIP: {CLASS_NAMES[clip_pred]:10s} ({clip_conf*100:.1f}% confidence)")
    
    # Agreement check
    if medclip_pred == clip_pred:
        print(f"  ✓ Models agree: {CLASS_NAMES[medclip_pred]}")
    else:
        print(f"  ⚠ Models disagree!")
    
    # Save results
    results.append({
        'image': img_file,
        'medclip_prediction': CLASS_NAMES[medclip_pred],
        'medclip_confidence': f"{medclip_conf*100:.1f}%",
        'clip_prediction': CLASS_NAMES[clip_pred],
        'clip_confidence': f"{clip_conf*100:.1f}%",
        'agreement': 'Yes' if medclip_pred == clip_pred else 'No'
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/inference_results.csv', index=False)

print("\n" + "=" * 80)
print("✓ Inference Complete!")
print("=" * 80)
print(f"\nResults saved to: results/inference_results.csv")
print(f"\nSummary:")
print(f"  Total images: {len(results)}")
print(f"  Agreement: {sum(1 for r in results if r['agreement'] == 'Yes')}/{len(results)}")
print("\n" + "=" * 80)
