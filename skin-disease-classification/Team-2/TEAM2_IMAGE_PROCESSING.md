# Team 2: Image Processing

## Overview
This pipeline prepares the HAM10000 images using the Team 1 lesion-level split and stores the processed output in a separate folder without modifying the original raw dataset. The processing flow is designed to be compatible with a later deep-learning pipeline and keeps train/validation/test assignments unchanged.

## Inputs
- Team 1 manifest: `../Team-1/TEAM1_OUTPUT/team1_dataset_split.csv`
- Original HAM10000 metadata file: `HAM10000_metadata.csv`
- Original HAM10000 image files in the dataset root

## Dataset loading
1. Read the Team 1 split manifest from `team1_dataset_split.csv`.
2. Validate that required columns exist: `image_id`, `lesion_id`, `dx`, and `split`.
3. Confirm there is no overlap in `lesion_id` across train, validation, and test.
4. Load each image by matching its `image_id` to the actual file stem in the dataset directory.

## Corrupted image handling
The processing script checks every image before resizing. If an image is unreadable, missing, or fails the OpenCV read step, it is recorded in `corrupted_images.csv` and excluded from the processed output. This prevents invalid image files from breaking the training pipeline.

## Resize and color handling
- All images are resized to `224 x 224` using OpenCV.
- Images are read with OpenCV in BGR order using `cv2.imread(..., cv2.IMREAD_COLOR)`.
- BGR is converted to RGB when the image is loaded for visualization and processing.
- Saved processed images remain in valid uint8 form with values in the range `0..255`.
- Pixel values are not normalized at save time.

## Normalization policy
The pipeline follows the recommended preprocessing scheme:
- Save processed images as uint8 `0..255` arrays.
- Apply `rescale=1/255` only inside the data loader or generator.
- Do not normalize the saved PNG output a second time.

This avoids double-normalization and keeps the processed dataset compatible with standard Keras/TensorFlow or PyTorch image loading pipelines.

## Training augmentation
Data augmentation is applied only to the training split:
- random rotation
- horizontal flip
- vertical flip
- zoom/scale adjustment with safe resizing

Validation and test images receive only preprocessing (resize and RGB conversion) and no augmentation.

## Output structure
The processed dataset is written under a dedicated directory, for example:

- `TEAM2_OUTPUT/`
  - `train/`
    - `akiec/`
    - `bcc/`
    - ...
  - `validation/`
  - `test/`
  - `statistics/`
    - `processing_statistics.csv`
    - `class_wise_statistics.csv`
    - `corrupted_images.csv`
  - `visualizations/`
    - `before_after_samples.png`
  - `metadata/`
    - `team1_dataset_split.csv`
  - `processing_summary.json`
  - `processed_images_validation.csv`

## Validation checks
Each processed image is checked to confirm:
- it opens successfully with OpenCV
- the size is `224 x 224`
- dtype is `uint8`
- pixel values stay between `0` and `255`
- the output is ready for Team 3 model input

## Processing statistics
The pipeline records:
- train image count
- validation image count
- test image count
- corrupted image count
- class-wise counts per split

## Documentation and handover notes
The final Team 2 artifact is designed to hand off a clean, consistent image dataset to Team 3. The deep-learning model code is intentionally not included here. Team 3 should consume the processed images and the Team 1 split metadata without creating a new split or retraining the dataset.

## Important implementation rules
- Do not overwrite the original HAM10000 dataset.
- Do not create a new train/validation/test split.
- Do not augment validation or test data.
- Do not normalize saved images twice.
- Keep the processed dataset in a separate output folder.
