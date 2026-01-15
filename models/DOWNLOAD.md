# Trained Models - Download Instructions

The trained models are too large for GitHub (>100MB). Please download them from Google Drive:

## 📥 Download Links

### Option 1: Complete Inference Package (Recommended)
**Download the ready-to-use inference package with both models:**
- 📦 **inference.zip** - Contains both models, sample images, and results
- 🔗 **Google Drive Link**: `[TO BE ADDED - Upload and share the link]`
- 📊 Size: ~450 MB
- ✅ Includes: Both trained models, 5 sample images, inference scripts, results

### Option 2: Individual Models

**MedCLIP Model** (86.75% accuracy)
- 📄 File: `medclip_best.bin`
- 🔗 **Google Drive Link**: `[TO BE ADDED]`
- 📊 Size: 109 MB
- 📍 Place in: `models/` or `inference/models/`

**OpenAI CLIP Model** (85.54% accuracy)
- 📄 File: `clip_best.pth`
- 🔗 **Google Drive Link**: `[TO BE ADDED]`
- 📊 Size: 339 MB
- 📍 Place in: `models/` or `inference/models/`

### Option 3: Pretrained Weights (Optional)

**MedCLIP Pretrained Weights**
- 📄 Files: `medclip-vit-pretrained.zip`, `pytorch_model.bin`
- 🔗 **Google Drive Link**: `[TO BE ADDED]`
- 📊 Size: ~1 GB
- 📍 Place in: `pretrained/medclip-vit/`
- ℹ️ Only needed if you want to retrain from scratch

## 🚀 Quick Start After Download

### For Inference Package:
```bash
# 1. Download inference.zip from Google Drive
# 2. Extract to project root
unzip inference.zip

# 3. Run inference
cd inference
python run_inference.py
```

### For Individual Models:
```bash
# 1. Download model files from Google Drive
# 2. Create directories
mkdir -p models inference/models

# 3. Place downloaded files
mv medclip_best.bin models/
mv clip_best.pth models/

# 4. Copy to inference folder
cp models/* inference/models/

# 5. Run inference
cd inference
python run_inference.py
```

## 📋 Model Details

| Model | Accuracy | Size | Use Case |
|-------|----------|------|----------|
| **MedCLIP** | 86.75% | 109 MB | Medical imaging (Recommended) |
| **OpenAI CLIP** | 85.54% | 339 MB | General purpose |

## ⚠️ Important Notes

- Models are required for inference
- Pretrained weights are optional (only for retraining)
- All models use PyTorch format (`.pth` or `.bin`)
- Compatible with PyTorch 2.11.0+ and CUDA 13.0+

## 🔧 Alternative: Train Your Own

If you prefer to train from scratch:
```bash
# Prepare dataset
python src/data/prepare_dataset.py

# Train MedCLIP
python src/train/train_medclip.py

# Train OpenAI CLIP
python src/train/train_clip.py
```

Training takes ~6 minutes on NVIDIA GB10 GPU.

---

**Need help?** Check the main [README.md](../README.md) for complete documentation.
