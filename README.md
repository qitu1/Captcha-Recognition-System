# CAPTCHA Recognition System

A CNN-based CAPTCHA recognition system using character segmentation and classification approach.

## Project Structure

```
├── dataset/                    # Original CAPTCHA images
│   ├── train/                 # Training images (9000+)
│   └── test/                  # Test images (1900+)
│
├── src/                        # Source notebooks (run in order)
│   ├── segmentation.ipynb     # Step 1: Detect character bounding boxes
│   ├── extract_and_normalize.ipynb  # Step 2: Extract and normalize characters
│   ├── baseline_model.ipynb     # Step 3: Train CNN and evaluate
│   └── improved_model.ipynb     # Step 4: Improve CNN and evaluate
│
├── Segmented_dataset/          # Processed data
│   ├── train_labels/          # YOLO format labels (bounding boxes)
│   ├── test_labels/           
│   ├── train_characters/      # Extracted character images (36 classes: 0-9, a-z)
│   ├── test_characters/       
│   └── cache/                 # Cached .npz files for fast loading
│
└── models/
    └── saved_models/          
        ├── cnn_character_recognition.keras  # Trained baseline model
        ├── best_autoencoder.pth # Improved model encoder
        ├── best_baseline_cnn.pth
        ├── best_finetune_cnn.pth # Improved finetuned model


segmentation_tokenization.ipynb/  #original segmentation file by Peidong
```

## Workflow

### 1. Character Segmentation (`segmentation.ipynb`)
- Input: CAPTCHA images
- Output: YOLO format bounding boxes for each character
- Method: Connected component analysis + morphological operations

### 2. Character Extraction (`extract_and_normalize.ipynb`)
- Input: Images + YOLO labels
- Output: Individual character images (32x32 RGB)
- Organizes characters into 36 folders (0-9, a-z)
- Creates `.npz` cache for fast loading

### 3. CNN Training & Evaluation (`baseline_model.ipynb`)
- Architecture: 3-layer CNN (32→64→128 filters)
- Training: Full training set (46k+ characters)
- Evaluation: Test set with two metrics:
  - **Character-level accuracy**: Individual character predictions
  - **String-level accuracy**: Complete CAPTCHA predictions (all characters must be correct)

### 4. Improved Model (`improved_model.ipynb`)
- **Two-stage training approach** for better performance:
  - **Stage 1 - Autoencoder Pretraining**: 
    - Learns robust feature representations from character images
    - Encoder-decoder architecture with 128-dim latent space
    - Helps model learn generalizable features for noisy/distorted characters
  - **Stage 2 - Fine-tuning with Augmentation**:
    - Transfers pretrained encoder weights to classifier
    - Adds data augmentation (rotation, shift, zoom, shear)
    - Uses Cosine Annealing learning rate scheduler
- **Performance**: 
  - Baseline accuracy: **84.72%**
  - Improved accuracy: **88.33%** (+3.61% improvement)
- **Key advantage**: Better generalization to distorted and noisy characters

## Quick Start

1. **Run notebooks in order**:
   ```
   segmentation.ipynb → extract_and_normalize.ipynb → baseline_model.ipynb
   ```

2. **Or use cached data** (skip steps 1-2):
   - If `Segmented_dataset/cache/` exists, directly run `baseline_model.ipynb`

3. **Load trained model**:
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('models/saved_models/cnn_character_recognition.keras')
   ```

## Model Performance

- **Parameters**: ~235K (lightweight)
- **Character Accuracy**: ~82% (on individual characters)
- **String Accuracy**: ~30-40% (complete CAPTCHA, all chars correct)
- **Training Time**: ~6-8 minutes (CPU)

## Key Features

- **Fast data loading**: `.npz` cache reduces loading time from 2 minutes to 3 seconds
- **Model interpretability**: Visualizes CNN feature maps to show what the model "sees"
- **Overfitting detection**: Automatic analysis of train/test gap
- **Two-level metrics**: Both character-level and string-level accuracy

## Requirements

```
tensorflow>=2.10.0
torch>=2.0.0
torchvision
opencv-python
numpy
matplotlib
seaborn
scikit-learn
tqdm
```

**Note**: 
- `tensorflow` is required for baseline model (`baseline_model.ipynb`)
- `torch` and `torchvision` are required for improved model (`improved_model.ipynb`)

## Notes

- CAPTCHA format: Variable length (4-8 characters), alphanumeric (0-9, a-z)
- Image size: Original varies, normalized to 32x32 for CNN input
- Character extraction relies on YOLO bounding boxes from segmentation step
