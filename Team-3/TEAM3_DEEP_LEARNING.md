# Team 3 Deep Learning

## Overview
This Team 3 pipeline was completed using the available repository data only. The model was trained and verified on the Team 2 processed sample/gallery dataset, which contains 68 images total: 54 train, 7 validation, and 7 test. This is not the full HAM10000 dataset and should not be interpreted as full-dataset clinical performance.

## Team 2 input
The Team 2 processed data is located at [skin-disease-classification/Team-2/processed_data](skin-disease-classification/Team-2/processed_data). It contains the split folders and all required metadata.

The Team 2 contract requires:
- RGB images
- uint8 dtype
- values in the range 0–255
- no double normalization during saving
- a single rescale=1/255 when loading into TensorFlow/Keras

## Dataset counts
- train: 54
- validation: 7
- test: 7
- total: 68

## Classes
The actual class mapping in the repository is:
- akiec -> 0
- bcc -> 1
- bkl -> 2
- df -> 3
- mel -> 4
- nv -> 5
- vasc -> 6

## Image format and normalization
The saved Team 2 images are RGB uint8 arrays in the 0–255 range. Team 3 uses a single rescale of 1/255 in the ImageDataGenerator. No additional normalization is performed on disk.

## Augmentation
Training augmentation was applied only to the training split and was kept moderate to prevent overfitting on the small dataset. Augmentation included:
- rotation
- width/height shift
- horizontal flip
- vertical flip
- zoom

Validation and test data were not augmented.

## CNN architecture
The model in [Team-3/team3-deep-learning/model.py](Team-3/team3-deep-learning/model.py) is a compact CNN with:
- Conv2D blocks
- BatchNormalization
- ReLU activations
- MaxPooling2D
- GlobalAveragePooling2D
- Dropout
- Dense softmax output

The output layer size matches the seven classes in the class mapping.

## Training configuration
- optimizer: Adam
- learning rate: 1e-3
- loss: categorical_crossentropy
- batch size: 8
- epochs: 10
- callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- class weights: computed from the training data only

## Actual training results
The model was trained on the available 68-image dataset and produced the following actual metrics:
- epochs completed: 10
- best epoch: 6
- train accuracy: 0.5185
- val accuracy: 0.2857
- train loss: 1.3494
- val loss: 1.9467
- best val accuracy: 0.2857
- best val loss: 1.9221

These metrics are unstable because the dataset is very small and should not be treated as full-dataset generalization.

## Saved model artifacts
The final model and best checkpoint were saved under:
- [Team-3/team3-deep-learning/models/best_model_20260831_230834.keras](Team-3/team3-deep-learning/models/best_model_20260831_230834.keras)
- [Team-3/team3-deep-learning/models/final_model_20260831_230834.keras](Team-3/team3-deep-learning/models/final_model_20260831_230834.keras)

The recorded training artifacts are in:
- [Team-3/team3-deep-learning/results/training_history.csv](Team-3/team3-deep-learning/results/training_history.csv)
- [Team-3/team3-deep-learning/results/training_history.json](Team-3/team3-deep-learning/results/training_history.json)
- [Team-3/team3-deep-learning/results/training_accuracy.png](Team-3/team3-deep-learning/results/training_accuracy.png)
- [Team-3/team3-deep-learning/results/training_loss.png](Team-3/team3-deep-learning/results/training_loss.png)
- [Team-3/team3-deep-learning/model_metadata.json](Team-3/team3-deep-learning/model_metadata.json)

## Verification
The saved model was reloaded successfully and a real prediction was performed on a dataset image. The model output contained valid finite probabilities and a valid class index mapping to the repository’s class list.

The test split of 7 images was also passed through the model successfully. The actual test-set inference check produced predictions for all 7 images, but the results reflect the small sample dataset only.

## Limitation
Training was completed using the available 68-image sample/gallery dataset and not the full HAM10000 dataset. This is a repository-valid Team 3 completion, but it is not a full HAM10000-trained model.
