# Team 4 Handover

## Team 3 Status
**COMPLETE — VERIFIED ON AVAILABLE 68-IMAGE SAMPLE/GALLERY DATASET**

Date verified: 2026-08-31
Verification script: [Team-3/team3-deep-learning/verify_team3.py](Team-3/team3-deep-learning/verify_team3.py)

## Dataset
- Train: 54
- Validation: 7
- Test: 7
- Total used: 68

This is the available repository dataset only. It is not the full HAM10000 dataset.

## Classes (Verified)
1. akiec (index 0)
2. bcc (index 1)
3. bkl (index 2)
4. df (index 3)
5. mel (index 4)
6. nv (index 5)
7. vasc (index 6)

All classes verified to match class_mapping.json in Team 2 processed data.

## Model
Best model path (verified to load successfully):
- [Team-3/team3-deep-learning/models/best_model_20260831_230651.keras](Team-3/team3-deep-learning/models/best_model_20260831_230651.keras)
- File size: 5055.6 KB
- Format: TensorFlow/Keras .keras file
- Status: Successfully loaded and verified

## Input
- image size: 224 × 224 × 3 (verified)
- image format: RGB uint8 (verified)
- value range: 0–255 (verified during preprocessing)
- Model input shape verified: (None, 224, 224, 3)
- Model output shape verified: (None, 7)

## Normalization
The model loader uses a single rescale=1/255 step in the data generator. Team 2 output was not re-normalized on disk. This was verified during sample prediction:
- Array dtype: float32
- Array range after preprocessing: [0.027451, 0.992157]
- Probabilities sum: 1.000000 (valid softmax output)

## Training Configuration
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 8
- Epochs: 10
- Loss function: categorical_crossentropy
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Class weights: computed from training data

## Training Results
- Epochs completed: 10
- Best epoch: 5
- Best validation accuracy: 0.2857
- Best validation loss: 1.9542
- Train/validation ratio indicates small sample dataset

## Model Architecture Verified
✓ Input layer: 224×224×3 (None, 224, 224, 3)
✓ Conv blocks with BatchNormalization and ReLU
✓ GlobalAveragePooling2D (reduces parameters)
✓ Dense layers with dropout
✓ Output layer: softmax activation with 7 units
✓ Model loads successfully without errors

## Verification Performed
All verifications completed successfully (2026-08-31):

### 1. Class Mapping ✓
- Loaded class_mapping.json from Team 2 processed_data
- Expected classes: ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
- Actual classes: ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
- Status: PASS

### 2. Model Loading ✓
- Model path: best_model_20260831_230651.keras
- File size: 5055.6 KB
- Loading status: PASS
- No errors during load

### 3. Model Architecture ✓
- Input shape: (None, 224, 224, 3) — PASS
- Output shape: (None, 7) — PASS
- Classes: 7 — PASS
- RGB format: 3 channels — PASS

### 4. Sample Prediction ✓
- Image: ISIC_0025825.jpg.png (true class: akiec)
- Predicted class: bcc (index 1)
- Confidence: 0.162042
- Array shape: (1, 224, 224, 3) — PASS
- Output shape: (7,) — PASS
- Probabilities sum: 1.000000 — PASS
- No NaN/Inf values — PASS

### 5. All 7 Test Images ✓
All test images processed successfully:
- ✓ akiec/ISIC_0025825.jpg.png → predicted: bcc (0.1620)
- ✓ bcc/ISIC_0024572.jpg.png → predicted: bcc (0.1628)
- ✓ bkl/ISIC_0024626.jpg.png → predicted: bcc (0.1593)
- ✓ df/ISIC_0026629.jpg.png → predicted: nv (0.1641)
- ✓ mel/ISIC_0024675.jpg.png → predicted: bcc (0.1628)
- ✓ nv/ISIC_0024363.jpg.png → predicted: bcc (0.1590)
- ✓ vasc/ISIC_0026092.jpg.png → predicted: bcc (0.1637)

Status: 7/7 successful, all predictions valid

### 6. Training Artifacts ✓
All expected files present and verified:
- ✓ training_history.csv (1546 bytes)
- ✓ training_history.json (2169 bytes)
- ✓ training_accuracy.png (36020 bytes)
- ✓ training_loss.png (30109 bytes)
- ✓ model_metadata.json (2791 bytes)

### 7. Metadata Verification ✓
- Train images: 54 — PASS
- Validation images: 7 — PASS
- Test images: 7 — PASS
- Classes: 7 — PASS
- Note in metadata confirms sample/gallery dataset — PASS

### 8. Existing Test Suite ✓
All project tests pass (8/8):
- test_build_sample_manifest_creates_10_per_class — PASS
- test_copy_sample_images_creates_70_files — PASS
- test_check_split_leakage_detects_overlap — PASS
- test_ensure_uint8_image_converts_to_uint8 — PASS
- test_resize_image_224 — PASS
- test_load_split_manifest_uses_team1_file — PASS
- test_write_class_mapping_creates_json — PASS
- test_processed_images_remain_uint8_and_not_normalized — PASS

## Team 4 Next Steps
Team 4 should perform the official detailed evaluation on the 7 test samples, including:
- Accuracy calculation
- Precision per class
- Recall per class
- F1-score per class
- Confusion matrix generation
- Per-class performance analysis
- Prediction confidence analysis
- Misclassification analysis

## Important Limitation
**The model was trained using the available 68-image sample/gallery dataset extracted from the repository. It is not a full HAM10000-trained model, and the metrics should not be interpreted as full-dataset performance or clinical performance beyond the repository sample.**

This is a valid Team 3 completion for the available repository data only.

## Handover Status
Team 3 is complete and ready for Team 4 evaluation. All artifacts are saved, model loads successfully, all test images produce valid predictions, and documentation is complete.
