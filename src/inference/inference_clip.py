"""
Inference Script for OpenAI CLIP Toxoplasmosis Classifier

Usage:
    python inference_clip.py --image path/to/image.jpg
    python inference_clip.py --image path/to/image.jpg --checkpoint path/to/model.pth
"""

import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import clip

from clip_classifier import CLIPClassifier

# Class names
CLASS_NAMES = ['healthy', 'active', 'inactive']
NUM_CLASSES = 3

# Model configuration (must match training)
MLP_CONFIG = {
    'hidden_dims': [512, 256],
    'dropout': 0.4,
    'freeze_backbone': False,
}

def load_model(checkpoint_path, device='cuda'):
    """Load trained CLIP model from checkpoint"""
    print(f"Loading CLIP model...")
    
    # Load CLIP base model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    print("✓ Loaded CLIP ViT-B/32")
    
    # Create classifier
    model = CLIPClassifier(
        clip_model=clip_model,
        num_classes=NUM_CLASSES,
        input_dim=512,  # CLIP ViT-B/32 outputs 512-dim features
        hidden_dims=MLP_CONFIG['hidden_dims'],
        dropout=MLP_CONFIG['dropout'],
        mode='multiclass',
        freeze_backbone=MLP_CONFIG['freeze_backbone']
    )
    
    # Load trained weights
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded trained weights from: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path):
    """Preprocess image for CLIP model"""
    # CLIP's standard preprocessing (no augmentation for inference)
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),  # CLIP's normalization
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    
    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict(model, image_tensor, device='cuda'):
    """Make prediction on preprocessed image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor, return_loss=False)
        logits = outputs['logits']
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='OpenAI CLIP Toxoplasmosis Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/toxoplasmosis_clip_mlp/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("OpenAI CLIP Toxoplasmosis Classifier - Inference")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    # Load model
    model = load_model(args.checkpoint, device=args.device)
    
    # Preprocess image
    print("Preprocessing image...")
    image_tensor = preprocess_image(args.image)
    
    # Make prediction
    print("Making prediction...")
    predicted_class, confidence, probabilities = predict(model, image_tensor, device=args.device)
    
    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"\n🎯 Predicted Class: {CLASS_NAMES[predicted_class].upper()}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"\n📈 Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        bar = "█" * int(prob * 50)
        print(f"  {class_name:10s}: {prob:.2%} {bar}")
    print("=" * 80)

if __name__ == "__main__":
    main()
