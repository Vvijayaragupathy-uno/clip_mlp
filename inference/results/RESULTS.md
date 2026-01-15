# Model Performance Results

## MedCLIP Model (86.75% Accuracy)

### Classification Report
```
              precision    recall  f1-score   support

     healthy     0.7714    1.0000    0.8710        27
      active     1.0000    0.7222    0.8387        18
    inactive     0.9143    0.8421    0.8767        38

    accuracy                         0.8675        83
   macro avg     0.8952    0.8548    0.8621        83
weighted avg     0.8864    0.8675    0.8666        83
```

### Confusion Matrix
```
[[27  0  0]   <- Healthy: 27/27 correct (100%)
 [ 2 13  3]   <- Active: 13/18 correct (72.2%)
 [ 6  0 32]]  <- Inactive: 32/38 correct (84.2%)
```

### Key Insights
- **Best at**: Detecting healthy eyes (100% recall)
- **Challenging**: Active toxoplasmosis (72.2% recall)
- **Overall**: Strong performance across all classes

---

## OpenAI CLIP Model (85.54% Accuracy)

### Classification Report
```
              precision    recall  f1-score   support

     healthy     0.8966    0.9630    0.9286        27
      active     0.8462    0.6111    0.7097        18
    inactive     0.8293    0.8947    0.8608        38

    accuracy                         0.8554        83
   macro avg     0.8573    0.8229    0.8330        83
weighted avg     0.8548    0.8554    0.8501        83
```

### Confusion Matrix
```
[[26  0  1]   <- Healthy: 26/27 correct (96.3%)
 [ 1 11  6]   <- Active: 11/18 correct (61.1%)
 [ 2  2 34]]  <- Inactive: 34/38 correct (89.5%)
```

### Key Insights
- **Best at**: Detecting inactive cases (89.5% recall)
- **Challenging**: Active toxoplasmosis (61.1% recall)
- **Overall**: Balanced performance

---

## Model Comparison

| Metric | MedCLIP | OpenAI CLIP | Winner |
|--------|---------|-------------|--------|
| **Overall Accuracy** | 86.75% | 85.54% | MedCLIP |
| **Healthy Recall** | 100% | 96.3% | MedCLIP |
| **Active Recall** | 72.2% | 61.1% | MedCLIP |
| **Inactive Recall** | 84.2% | 89.5% | CLIP |
| **Model Size** | 109 MB | 339 MB | MedCLIP |

## Recommendation

**Use MedCLIP** for:
- Medical image analysis
- When detecting healthy vs diseased is critical
- Resource-constrained environments (smaller model)

**Use OpenAI CLIP** for:
- General-purpose applications
- When you need better inactive case detection
- Transfer learning to other domains

---

## Sample Predictions

Place your inference results here after running:
```bash
python inference_medclip.py --image sample_images/test.jpg
python inference_clip.py --image sample_images/test.jpg
```

Results will be automatically saved to this folder.
