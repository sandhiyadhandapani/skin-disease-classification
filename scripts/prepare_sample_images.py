from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def build_sample_manifest(metadata: pd.DataFrame, max_per_class: int = 10) -> pd.DataFrame:
    if metadata.empty:
        raise ValueError("Metadata DataFrame is empty.")
    if "dx" not in metadata.columns or "image_id" not in metadata.columns:
        raise ValueError("Metadata must contain 'dx' and 'image_id' columns.")

    valid = metadata[metadata["dx"].isin(CLASS_ORDER)].copy()
    if valid.empty:
        raise ValueError("No valid HAM10000 disease classes found in the provided metadata.")

    selected_rows = []
    for disease in CLASS_ORDER:
        class_rows = valid[valid["dx"] == disease].drop_duplicates(subset=["image_id"]).reset_index(drop=True)
        if len(class_rows) < max_per_class:
            raise ValueError(f"Class '{disease}' has fewer than {max_per_class} available images.")
        chosen = class_rows.iloc[:max_per_class].copy()
        chosen["class_name"] = disease
        chosen["sample_index"] = range(1, len(chosen) + 1)
        selected_rows.append(chosen)

    result = pd.concat(selected_rows, ignore_index=True)
    result = result[["image_id", "dx", "class_name", "sample_index"]]
    result = result.sort_values(["dx", "sample_index"]).reset_index(drop=True)
    return result


def copy_sample_images(selected_manifest: pd.DataFrame, dataset_dir: Path, output_dir: Path) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for disease in CLASS_ORDER:
        disease_dir = output_dir / disease
        disease_dir.mkdir(parents=True, exist_ok=True)
        disease_rows = selected_manifest[selected_manifest["dx"] == disease].reset_index(drop=True)
        for index, row in disease_rows.iterrows():
            image_id = str(row["image_id"])
            candidates = []
            for suffix in [".jpg", ".jpeg", ".png"]:
                candidate = Path(dataset_dir) / f"{image_id}{suffix}"
                if candidate.exists():
                    candidates.append(candidate)
            if not candidates:
                raise FileNotFoundError(f"Sample image not found for {image_id} in {dataset_dir}")
            source = candidates[0]
            target = disease_dir / f"{index + 1:02d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 70 reference sample images for the HAM10000 skin disease gallery.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Root path of the HAM10000 dataset")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("sample_images"), help="Directory to store the 70 sample images")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    selected = build_sample_manifest(metadata)
    copied = copy_sample_images(selected, args.dataset_dir, args.output_dir)
    print(f"Selected: {len(selected)} images across 7 classes")
    print(f"Copied: {copied} files to {args.output_dir}")


if __name__ == "__main__":
    main()
