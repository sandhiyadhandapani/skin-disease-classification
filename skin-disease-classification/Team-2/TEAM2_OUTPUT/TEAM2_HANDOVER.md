# Team 2 Dataset Handover

## Processed dataset location
The processed Team 2 dataset is located at: D:\Techtrio - intern\2nd major project\skin-disease-classification\skin-disease-classification\Team-2\TEAM2_OUTPUT

## Split locations
- Train: D:\Techtrio - intern\2nd major project\skin-disease-classification\skin-disease-classification\Team-2\TEAM2_OUTPUT\train
- Validation: D:\Techtrio - intern\2nd major project\skin-disease-classification\skin-disease-classification\Team-2\TEAM2_OUTPUT\validation
- Test: D:\Techtrio - intern\2nd major project\skin-disease-classification\skin-disease-classification\Team-2\TEAM2_OUTPUT\test

## Image specification
- Image size: 224 x 224
- Color format: RGB
- Normalization strategy: saved as uint8 RGB PNGs in the 0..255 range; do not apply a second 1/255 rescale in Team 3
- Augmentation: random flips, rotation, and zoom are applied only to training images; validation/test images are deterministic and unaugmented

## Split counts
- Train: 56
- Validation: 7
- Test: 5
- Failed/corrupted: 9947

## Label structure
Processed images are saved under class folders using the HAM10000 dx codes: akiec, bcc, bkl, df, mel, nv, vasc.

## Team 3 loading guidance
Use the split folder root and load the class folders as labels, or use a custom generator. Do not create a new split or alter the Team 1 train/validation/test manifest.

## Do not do this again
- Do not normalize the saved PNG files before loading.
- Do not apply rescale=1/255 or Rescaling(1.0 / 255.0) to these processed images.
- Do not mix train/validation/test images or reuse images across splits.

## Leakage check
The Team 1 manifest was preserved as the authoritative split and checked for lesion_id leakage before processing.
