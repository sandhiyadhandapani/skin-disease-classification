# Team 1 Dataset Handover

## Dataset summary
HAM10000 contains **10,015 metadata records/images**, **7 classes**, and **7,470 unique lesion IDs**. The raw archive remains unchanged.

## Class distribution
See `class_distribution.csv` for the complete raw distribution and `split_class_distribution.csv` for distribution within each split. The split was generated with seed **42**, using whole `lesion_id` groups and target ratios of 80% train, 10% validation, and 10% test.

## Data quality and duplicates
Missing values, duplicate IDs/rows, repeated lesions, and image/metadata reconciliation are documented in `TEAM1_DATASET_RESEARCH.md` and the corresponding CSV reports. Repeated lesion IDs were deliberately kept together during splitting.

## Split counts

| Split | Images | Lesions |
|---|---:|---:|
| Train | 8,012 | 5,501 |
| Validation | 1,000 | 983 |
| Test | 1,003 | 986 |

## Leakage check
**PASSED: no lesion_id appears in more than one split.** See `split_leakage_check.csv`.

## Files created
- `team1_dataset_analysis.py`: reproducible analysis and split script.
- `team1_dataset_split.csv`: handover manifest with `image_id`, `lesion_id`, `dx`, and `split`.
- `split_class_distribution.csv`: train/validation/test class counts and percentages.
- `class_distribution.csv`, `class_mapping.csv`, quality CSV reports, and PNG graphs.
- `TEAM1_DATASET_RESEARCH.md`: dataset research report.

## What Team 2 needs to use
Use `team1_dataset_split.csv` as the authoritative manifest. Filter by `split` and load the matching `image_id` from the original HAM10000 image directories. Keep the provided train/validation/test assignment and labels; do not create a new image-level split.

## What Team 3 needs to use
Use the same manifest and split labels for evaluation, reporting, and any application integration. The test rows must remain untouched until final evaluation, and the raw image files should be read from the original archive.
