from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

SEED = 42
SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def find_dataset() -> Path:
    candidates = [
        Path.home() / "Downloads" / "archive (1)",
        Path.home() / "Downloads" / "HAM10000",
        Path.home() / "Downloads",
    ]
    for candidate in candidates:
        if (candidate / "HAM10000_metadata.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate HAM10000_metadata.csv in common Downloads locations")


def image_files(dataset_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def write_class_distribution(metadata: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    counts = metadata["dx"].value_counts().sort_index()
    report = pd.DataFrame({"dx": counts.index, "disease_name": [CLASS_NAMES.get(code, "Unknown") for code in counts.index], "image_count": counts.values})
    report["percentage"] = report["image_count"] / len(metadata) * 100
    report.to_csv(output_dir / "class_distribution.csv", index=False)
    return report


def create_class_graph(report: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(report["dx"], report["image_count"], color="#1f6f78")
    ax.set_title("HAM10000 Class Distribution")
    ax.set_xlabel("Disease code")
    ax.set_ylabel("Number of images")
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(fig)


def create_sample_graph(metadata: pd.DataFrame, files: dict[str, Path], output_dir: Path) -> None:
    classes = sorted(metadata["dx"].dropna().unique())
    figure, axes = plt.subplots(1, len(classes), figsize=(3 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]
    for axis, disease_code in zip(axes, classes):
        row = metadata[metadata["dx"] == disease_code].sort_values("image_id").iloc[0]
        path = files.get(row["image_id"])
        if path is not None:
            with Image.open(path) as image:
                axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        axis.set_title(f"{disease_code}\n{CLASS_NAMES.get(disease_code, 'Unknown')}", fontsize=9)
        axis.axis("off")
    figure.suptitle("One HAM10000 Example Per Disease Class")
    figure.tight_layout()
    figure.savefig(output_dir / "sample_images_by_class.png", dpi=180)
    plt.close(figure)


def create_quality_reports(metadata: pd.DataFrame, files: dict[str, Path], output_dir: Path) -> dict[str, int]:
    missing = metadata.isna().sum().rename("missing_count").to_frame()
    missing["missing_percentage"] = missing["missing_count"] / len(metadata) * 100
    missing.to_csv(output_dir / "metadata_missing_values.csv")

    duplicate_image_ids = metadata[metadata.duplicated("image_id", keep=False)].sort_values("image_id")
    duplicate_image_ids.to_csv(output_dir / "duplicate_image_ids.csv", index=False)
    duplicate_rows = metadata[metadata.duplicated(keep=False)].sort_values(list(metadata.columns))
    duplicate_rows.to_csv(output_dir / "duplicate_rows.csv", index=False)
    repeated_lesions = metadata[metadata.duplicated("lesion_id", keep=False)].sort_values("lesion_id")
    repeated_lesions.to_csv(output_dir / "repeated_lesion_ids.csv", index=False)

    metadata_ids = set(metadata["image_id"].dropna())
    file_ids = set(files)
    metadata_without_files = sorted(metadata_ids - file_ids)
    files_without_metadata = sorted(file_ids - metadata_ids)
    pd.DataFrame({"image_id": metadata_without_files}).to_csv(output_dir / "metadata_images_without_files.csv", index=False)
    pd.DataFrame({"image_id": files_without_metadata}).to_csv(output_dir / "image_files_without_metadata.csv", index=False)
    pd.DataFrame([
        {"check": "metadata image IDs without image files", "count": len(metadata_without_files)},
        {"check": "image files without metadata records", "count": len(files_without_metadata)},
    ]).to_csv(output_dir / "image_metadata_comparison.csv", index=False)

    summary = pd.DataFrame([
        {"finding": "metadata_records", "count": len(metadata)},
        {"finding": "unique_image_ids", "count": metadata["image_id"].nunique(dropna=True)},
        {"finding": "duplicate_image_id_records", "count": len(duplicate_image_ids)},
        {"finding": "duplicate_rows", "count": len(duplicate_rows)},
        {"finding": "unique_lesion_ids", "count": metadata["lesion_id"].nunique(dropna=True)},
        {"finding": "records_with_repeated_lesion_id", "count": len(repeated_lesions)},
    ])
    summary.to_csv(output_dir / "duplicate_findings.csv", index=False)

    column_rows = []
    for column in ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"]:
        column_rows.append({
            "column": column,
            "dtype": str(metadata[column].dtype),
            "unique_values": metadata[column].nunique(dropna=True),
            "missing_count": int(metadata[column].isna().sum()),
            "top_values": "; ".join(f"{key}={value}" for key, value in metadata[column].value_counts(dropna=False).head(10).items()),
        })
    pd.DataFrame(column_rows).to_csv(output_dir / "column_summary.csv", index=False)
    pd.DataFrame({"dx": sorted(CLASS_NAMES), "disease_name": [CLASS_NAMES[code] for code in sorted(CLASS_NAMES)]}).to_csv(output_dir / "class_mapping.csv", index=False)
    return {"duplicate_image_id_records": len(duplicate_image_ids), "duplicate_rows": len(duplicate_rows), "repeated_lesion_records": len(repeated_lesions), "metadata_without_files": len(metadata_without_files), "files_without_metadata": len(files_without_metadata)}


def lesion_level_split(metadata: pd.DataFrame) -> pd.DataFrame:
    group_sizes = metadata.groupby(["dx", "lesion_id"], dropna=False).size().reset_index(name="image_count")
    assignments = []
    for disease_code, groups in group_sizes.groupby("dx", sort=True):
        groups = groups.sample(frac=1, random_state=SEED).sort_values("image_count", ascending=False, kind="stable")
        targets = {split: len(metadata[metadata["dx"] == disease_code]) * ratio for split, ratio in SPLIT_RATIOS.items()}
        current = {split: 0 for split in SPLIT_RATIOS}
        for row in groups.itertuples(index=False):
            split = min(SPLIT_RATIOS, key=lambda name: (current[name] - targets[name], current[name], name))
            assignments.append({"lesion_id": row.lesion_id, "dx": disease_code, "split": split})
            current[split] += row.image_count
    assignment_frame = pd.DataFrame(assignments)
    result = metadata.merge(assignment_frame, on=["lesion_id", "dx"], how="left", validate="many_to_one")
    return result[["image_id", "lesion_id", "dx", "split"]]


def create_split_reports(metadata: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, bool]:
    split = lesion_level_split(metadata)
    split.to_csv(output_dir / "team1_dataset_split.csv", index=False)
    report = split.groupby(["split", "dx"]).size().rename("image_count").reset_index()
    report["percentage_within_split"] = report.groupby("split")["image_count"].transform(lambda values: values / values.sum() * 100)
    report["disease_name"] = report["dx"].map(CLASS_NAMES)
    report = report[["split", "dx", "disease_name", "image_count", "percentage_within_split"]].sort_values(["split", "dx"])
    report.to_csv(output_dir / "split_class_distribution.csv", index=False)
    split_sets = {name: set(split.loc[split["split"] == name, "lesion_id"]) for name in SPLIT_RATIOS}
    leakage = any(split_sets[first] & split_sets[second] for first in split_sets for second in split_sets if first < second)
    pd.DataFrame([{"check": "lesion_id leakage between splits", "passed": not leakage}]).to_csv(output_dir / "split_leakage_check.csv", index=False)
    return split, not leakage


def write_documents(metadata: pd.DataFrame, class_report: pd.DataFrame, quality: dict[str, int], split: pd.DataFrame, leakage_free: bool, output_dir: Path) -> None:
    split_counts = split["split"].value_counts().to_dict()
    class_lines = "\n".join(f"| `{row.dx}` | {row.disease_name} | {row.image_count:,} | {row.percentage:.2f}% |" for row in class_report.itertuples())
    research = f"""# Team 1 Dataset Research

## Dataset overview
HAM10000 image dataset located at `{find_dataset()}`. The raw dataset was only read; it was not modified. Analysis used `HAM10000_metadata.csv` and JPG/PNG files in both image-part directories.

- Images found: **{len(image_files(find_dataset())):,}**
- Metadata records: **{len(metadata):,}**
- Unique lesion IDs: **{metadata['lesion_id'].nunique():,}**
- Disease classes: **{metadata['dx'].nunique()}**

## Classes and distribution

| Code | Disease name | Images | Percentage |
|---|---|---:|---:|
{class_lines}

## Metadata
The metadata columns are `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, and `localization`. Detailed per-column cardinality, type, missingness, and top values are in `column_summary.csv`; the code/name mapping is in `class_mapping.csv`.

## Data quality findings
- Missing metadata values are reported in `metadata_missing_values.csv`.
- Duplicate image IDs: **{quality['duplicate_image_id_records']:,} records**.
- Duplicate metadata rows: **{quality['duplicate_rows']:,}**.
- Records belonging to repeated lesion IDs: **{quality['repeated_lesion_records']:,}**. Repeated lesion IDs are expected in HAM10000 because a lesion can have multiple images.
- Metadata image IDs without files: **{quality['metadata_without_files']:,}**.
- Image files without metadata: **{quality['files_without_metadata']:,}**.

## Limitations
HAM10000 is class-imbalanced, contains multiple images of some lesions, and is not necessarily representative of all populations or clinical settings. Labels have different acquisition/confirmation types (`dx_type`), demographic fields contain missing values, and images may be correlated by patient or lesion. Results from this dataset should not be treated as clinical diagnosis without external validation.
"""
    (output_dir / "TEAM1_DATASET_RESEARCH.md").write_text(research, encoding="utf-8")

    split_rows = "\n".join(f"| {name.title()} | {int(split_counts.get(name, 0)):,} | {int(split[split['split'] == name]['lesion_id'].nunique()):,} |" for name in SPLIT_RATIOS)
    handover = f"""# Team 1 Dataset Handover

## Dataset summary
HAM10000 contains **{len(metadata):,} metadata records/images**, **{metadata['dx'].nunique()} classes**, and **{metadata['lesion_id'].nunique():,} unique lesion IDs**. The raw archive remains unchanged.

## Class distribution
See `class_distribution.csv` for the complete raw distribution and `split_class_distribution.csv` for distribution within each split. The split was generated with seed **{SEED}**, using whole `lesion_id` groups and target ratios of 80% train, 10% validation, and 10% test.

## Data quality and duplicates
Missing values, duplicate IDs/rows, repeated lesions, and image/metadata reconciliation are documented in `TEAM1_DATASET_RESEARCH.md` and the corresponding CSV reports. Repeated lesion IDs were deliberately kept together during splitting.

## Split counts

| Split | Images | Lesions |
|---|---:|---:|
{split_rows}

## Leakage check
**{'PASSED: no lesion_id appears in more than one split.' if leakage_free else 'FAILED: lesion_id leakage detected.'}** See `split_leakage_check.csv`.

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
"""
    (output_dir / "TEAM1_HANDOVER.md").write_text(handover, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir or find_dataset()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(dataset_dir / "HAM10000_metadata.csv")
    files = image_files(dataset_dir)
    class_report = write_class_distribution(metadata, output_dir)
    create_class_graph(class_report, output_dir)
    create_sample_graph(metadata, files, output_dir)
    quality = create_quality_reports(metadata, files, output_dir)
    split, leakage_free = create_split_reports(metadata, output_dir)
    write_documents(metadata, class_report, quality, split, leakage_free, output_dir)
    print(f"Dataset: {dataset_dir}")
    print(f"Images: {len(files)}; metadata records: {len(metadata)}; classes: {metadata['dx'].nunique()}")
    print(split["split"].value_counts().sort_index().to_string())
    print(f"Leakage check: {'PASSED' if leakage_free else 'FAILED'}")


if __name__ == "__main__":
    main()
