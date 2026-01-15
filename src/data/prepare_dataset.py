"""
Prepare Toxoplasmosis Dataset for MedCLIP Training
Converts the dataset to MedCLIP-compatible format
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from collections import Counter

# Paths - updated for new folder structure
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
dataset_dir = os.path.join(project_root, 'data', 'Ocular_Toxoplasmosis_Data_V3')
labels_file = os.path.join(dataset_dir, 'dataset_labels.csv')
images_dir = os.path.join(dataset_dir, 'images')
output_dir = os.path.join(project_root, 'data', 'processed')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the original labels
print("=" * 80)
print("Loading Toxoplasmosis Dataset")
print("=" * 80)
df = pd.read_csv(labels_file)

# Remove the last empty row if exists
df = df.dropna(subset=['Image_name'])

print(f"\nTotal images: {len(df)}")
print(f"\nClass distribution:")
print(df['Label'].value_counts())

# Create full image paths
df['imgpath'] = df['Image_name'].apply(lambda x: os.path.join(images_dir, x))

# Verify all images exist
missing_images = []
for idx, row in df.iterrows():
    if not os.path.exists(row['imgpath']):
        missing_images.append(row['Image_name'])

if missing_images:
    print(f"\n⚠ Warning: {len(missing_images)} images not found")
    print(f"First few missing: {missing_images[:5]}")
    # Remove missing images
    df = df[df['imgpath'].apply(os.path.exists)]
    print(f"Continuing with {len(df)} images")

# Create binary labels for each class (one-hot encoding)
# Classes: healthy, active, inactive
df['healthy'] = (df['Label'] == 'healthy').astype(int)
df['active'] = (df['Label'] == 'active').astype(int)
df['inactive'] = (df['Label'] == 'inactive').astype(int)

# Add subject_id (can be image name without extension)
df['subject_id'] = df['Image_name'].apply(lambda x: os.path.splitext(x)[0])

# Add empty report column (required by MedCLIP format)
df['report'] = ''

# Split into train and validation sets (80-20 split, stratified)
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['Label']
)

print("\n" + "=" * 80)
print("Dataset Split")
print("=" * 80)
print(f"\nTraining set: {len(train_df)} images")
print(train_df['Label'].value_counts())
print(f"\nValidation set: {len(val_df)} images")
print(val_df['Label'].value_counts())

# Prepare columns in MedCLIP format
# Required columns: imgpath, subject_id, report, and label columns
columns_to_save = ['imgpath', 'subject_id', 'report', 'healthy', 'active', 'inactive', 'Source']

# Save train and validation sets
train_output = os.path.join(output_dir, 'toxoplasmosis-train-meta.csv')
val_output = os.path.join(output_dir, 'toxoplasmosis-val-meta.csv')

train_df[columns_to_save].to_csv(train_output, index=True)
val_df[columns_to_save].to_csv(val_output, index=True)

print("\n" + "=" * 80)
print("Dataset Preparation Complete!")
print("=" * 80)
print(f"\n✓ Training data saved to: {train_output}")
print(f"✓ Validation data saved to: {val_output}")

# Create a summary file
summary = {
    'Total Images': len(df),
    'Training Images': len(train_df),
    'Validation Images': len(val_df),
    'Classes': ['healthy', 'active', 'inactive'],
    'Train Distribution': dict(train_df['Label'].value_counts()),
    'Val Distribution': dict(val_df['Label'].value_counts()),
}

print("\n" + "=" * 80)
print("Dataset Summary")
print("=" * 80)
for key, value in summary.items():
    print(f"{key}: {value}")

print("\n✓ Ready for MedCLIP training!")
