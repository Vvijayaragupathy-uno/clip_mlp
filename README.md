# CLIP Fine-tuning for Toxoplasmosis Classification

This project fine-tunes CLIP models (MedCLIP and OpenAI CLIP) for ocular toxoplasmosis classification.

## Project Structure

```
clip_mlp/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/                             # Dataset directory
│   ├── Ocular_Toxoplasmosis_Data_V3/ # Raw image data
│   └── processed/                    # Processed datasets
│       ├── toxoplasmosis-train-meta.csv
│       └── toxoplasmosis-val-meta.csv
│
├── models/                           # Trained models
│   ├── medclip_best.bin             # MedCLIP model (86.75% accuracy)
│   └── clip_best.pth                # OpenAI CLIP model (85.54% accuracy)
│
├── pretrained/                       # Pretrained weights
│   └── medclip-vit/                 # MedCLIP pretrained
│
├── src/                              # Source code
│   ├── models/                      # Model definitions
│   │   └── clip_classifier.py       # CLIP + MLP classifier
│   ├── data/                        # Data processing
│   │   └── prepare_dataset.py       # Dataset preparation
│   ├── train/                       # Training scripts
│   │   ├── train_medclip.py        # MedCLIP training
│   │   └── train_clip.py           # OpenAI CLIP training
│   └── inference/                   # Inference scripts
│       ├── inference_medclip.py
│       └── inference_clip.py
│
├── notebooks/                        # Jupyter notebooks
│   └── BiomedCLIP_Toxoplasmosis_classify.ipynb
│
├── logs/                            # Training logs
│   ├── training_medclip.log
│   └── training_clip.log
│
└── MedCLIP/                         # MedCLIP library (external)
```

## Dataset

- **Total Images**: 412
- **Training**: 329 images
- **Validation**: 83 images
- **Classes**: healthy, active, inactive toxoplasmosis

## Model Performance

| Model | Accuracy | Healthy | Active | Inactive |
|-------|----------|---------|--------|----------|
| **MedCLIP** | **86.75%** | 100% recall | 72.2% recall | 84.2% recall |
| **OpenAI CLIP** | **85.54%** | 96.3% recall | 61.1% recall | 89.5% recall |

## Installation

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### GPU Support (NVIDIA GB10)

For NVIDIA GB10 GPU (Blackwell architecture), install PyTorch nightly with CUDA 13.0:

```bash
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

## Usage

### 1. Prepare Dataset

```bash
python src/data/prepare_dataset.py
```

### 2. Train Models

**MedCLIP:**
```bash
python src/train/train_medclip.py
```

**OpenAI CLIP:**
```bash
python src/train/train_clip.py
```

### 3. Run Inference

**MedCLIP:**
```bash
python src/inference/inference_medclip.py --image path/to/image.jpg
```

**OpenAI CLIP:**
```bash
python src/inference/inference_clip.py --image path/to/image.jpg
```

### 4. Jupyter Notebook

```bash
jupyter notebook notebooks/BiomedCLIP_Toxoplasmosis_classify.ipynb
```

## GPU Compatibility

This project was developed and tested on:
- **GPU**: NVIDIA GB10 (Blackwell architecture)
- **CUDA**: 13.0
- **PyTorch**: 2.11.0.dev20260115+cu130 (nightly)

For older GPUs, use PyTorch stable releases with appropriate CUDA versions.

## Citation

If you use this code, please cite:
- MedCLIP: [MedCLIP Repository](https://github.com/RyanWangZf/MedCLIP)
- OpenAI CLIP: [CLIP Repository](https://github.com/openai/CLIP)

## License

This project uses code from MedCLIP and OpenAI CLIP. Please refer to their respective licenses.
