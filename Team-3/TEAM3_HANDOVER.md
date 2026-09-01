# Team 3 Handover

## 1. Team 3 Status
The training pipeline is prepared and aligned to the Team 2 preprocessing contract, but this repository does not contain the full HAM10000 dataset or a completed full-data training run. Therefore the honest status is: Team 2 preprocessing is verified and Team 3 is ready to train on the real data, but a full production training claim cannot be made without the complete dataset and a real training execution.

## 2. Team 2 Input Verified
The Team 2 artifact is stored under the repository at [skin-disease-classification/Team-2/processed_data](skin-disease-classification/Team-2/processed_data). It contains the processed train, validation, and test directories plus the metadata and validation files expected by Team 3.

The canonical Team 2 contract is documented in [skin-disease-classification/Team-2/TEAM2_IMAGE_PROCESSING.md](skin-disease-classification/Team-2/TEAM2_IMAGE_PROCESSING.md):
- saved images are RGB and uint8
- values stay in the range 0 to 255
- no second normalization is applied at save time
- the model loader applies a single rescale of 1/255
- train-only augmentation is applied upstream by Team 2

## 3. Dataset Counts
The verified sample-data run created the following split counts in [skin-disease-classification/Team-2/processed_data/processing_summary.json](skin-disease-classification/Team-2/processed_data/processing_summary.json):
- train: 54
- validation: 7
- test: 7
- corrupted images: 0

This is a sample-gallery-sized artifact and is not the full HAM10000 distribution.

## 4. Classes
The class mapping in [skin-disease-classification/Team-2/processed_data/class_mapping.json](skin-disease-classification/Team-2/processed_data/class_mapping.json) is:
- akiec
- bcc
- bkl
- df
- mel
- nv
- vasc

## 5. Model Architecture
The CNN definition in [Team-3/team3-deep-learning/model.py](Team-3/team3-deep-learning/model.py) uses:
- 4 convolution blocks
- batch normalization
- ReLU activation
- max pooling
- global average pooling
- dense head and softmax output

The class list and image shape match the Team 2 output contract.

## 6. Training Configuration
The training script in [Team-3/team3-deep-learning/train_cnn.py](Team-3/team3-deep-learning/train_cnn.py) uses:
- ImageDataGenerator with rescale=1/255
- class_mode="categorical"
- class names from the Team 2 class mapping
- Adam optimizer
- early stopping, checkpointing, and learning-rate reduction

Important rule: Team 3 does not re-save normalized images. It only rescales during loading.

## 7. Training Results
No full-data training metrics were produced in this repository because the full HAM10000 dataset is not present here, and no successful full training run is recorded. The model file and training script are ready, but the actual performance numbers must be produced on the real dataset when it is available.

## 8. Model Files
Existing Team 3 files are:
- [Team-3/team3-deep-learning/model.py](Team-3/team3-deep-learning/model.py)
- [Team-3/team3-deep-learning/train_cnn.py](Team-3/team3-deep-learning/train_cnn.py)

No final production model checkpoint was generated from a complete dataset run in this repo state.

## 9. Verification Results
The preprocessing contract is consistent with the Team 2 docs and the generated output files. The repository-specific validation checks are in [tests/test_team2_pipeline.py](tests/test_team2_pipeline.py) and [tests/test_sample_gallery_data.py](tests/test_sample_gallery_data.py).

## 10. Tests
The repo contains the regression checks for Team 2 processing and the sample-gallery setup, but the full training pipeline should not be described as completed unless it is executed on the complete dataset.

## 11. Git/GitHub Status
This work deliberately avoids altering the Team 1 data artifacts and keeps the Team 2 processed output available in the repository for Team 3. Generated outputs must be versioned in the repo, while environment folders such as caches and virtual-environments should remain local and uncommitted.

## 12. Team 4 Handover
When the full dataset becomes available and a real training run is completed, the Team 4 handover should include:
- final trained model file
- summary metrics
- confusion matrix and classification metrics
- exact class mapping used
- the processed output directory path

## 13. Remaining Problems
- The full HAM10000 dataset is not available in this repository state.
- No full, real-world training run is present.
- Therefore the repository is ready for the workflow, but the final model-performance claim remains pending until the full dataset is available and executed.
