# Team 1 Dataset Research

## Dataset overview
HAM10000 image dataset located at `C:\Users\SAMEER AHAMED\Downloads\archive (1)`. The raw dataset was only read; it was not modified. Analysis used `HAM10000_metadata.csv` and JPG/PNG files in both image-part directories.

- Images found: **10,015**
- Metadata records: **10,015**
- Unique lesion IDs: **7,470**
- Disease classes: **7**

## Classes and distribution

| Code | Disease name | Images | Percentage |
|---|---|---:|---:|
| `akiec` | Actinic keratoses and intraepithelial carcinoma | 327 | 3.27% |
| `bcc` | Basal cell carcinoma | 514 | 5.13% |
| `bkl` | Benign keratosis-like lesions | 1,099 | 10.97% |
| `df` | Dermatofibroma | 115 | 1.15% |
| `mel` | Melanoma | 1,113 | 11.11% |
| `nv` | Melanocytic nevi | 6,705 | 66.95% |
| `vasc` | Vascular lesions | 142 | 1.42% |

## Metadata
The metadata columns are `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, and `localization`. Detailed per-column cardinality, type, missingness, and top values are in `column_summary.csv`; the code/name mapping is in `class_mapping.csv`.

## Data quality findings
- Missing metadata values are reported in `metadata_missing_values.csv`.
- Duplicate image IDs: **0 records**.
- Duplicate metadata rows: **0**.
- Records belonging to repeated lesion IDs: **4,501**. Repeated lesion IDs are expected in HAM10000 because a lesion can have multiple images.
- Metadata image IDs without files: **0**.
- Image files without metadata: **0**.

## Limitations
HAM10000 is class-imbalanced, contains multiple images of some lesions, and is not necessarily representative of all populations or clinical settings. Labels have different acquisition/confirmation types (`dx_type`), demographic fields contain missing values, and images may be correlated by patient or lesion. Results from this dataset should not be treated as clinical diagnosis without external validation.
