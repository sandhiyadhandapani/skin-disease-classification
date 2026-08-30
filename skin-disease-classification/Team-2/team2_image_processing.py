from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_SIZE = (224, 224)
ALLOWED_SPLITS = ["train", "validation", "test"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def load_split_manifest(manifest_path: str | Path) -> pd.DataFrame:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    split_df = pd.read_csv(path)
    required = {"image_id", "lesion_id", "dx", "split"}
    missing = sorted(required - set(split_df.columns))
    if missing:
        raise ValueError(f"Split manifest is missing required columns: {missing}")
    split_df = split_df.copy()
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()
    return split_df


def check_split_leakage(split_df: pd.DataFrame) -> bool:
    split_sets = {
        split_name: set(split_df.loc[split_df["split"] == split_name, "lesion_id"].astype(str))
        for split_name in ALLOWED_SPLITS
    }
    for left_idx, left_split in enumerate(ALLOWED_SPLITS):
        for right_split in ALLOWED_SPLITS[left_idx + 1 :]:
            overlap = split_sets[left_split] & split_sets[right_split]
            if overlap:
                return False
    return True


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image array is None.")
    image = np.asarray(image)
    if image.size == 0:
        raise ValueError("Image array is empty.")
    if image.dtype == np.uint8:
        return image.copy()
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 255)
        return np.rint(image).astype(np.uint8)
    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8)
    return image.astype(np.uint8)


def find_dataset_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "HAM10000",
        Path(__file__).resolve().parents[2],
        Path.home() / "Downloads" / "HAM10000",
        Path.home() / "Downloads" / "archive (1)",
        Path.home() / "Downloads",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if (candidate / "HAM10000_metadata.csv").exists():
            return candidate.resolve()
        has_image_dirs = any(child.is_dir() and child.name.lower().endswith("lesion") for child in candidate.iterdir())
        has_image_files = any(child.is_file() and child.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} for child in candidate.rglob("*"))
        if has_image_dirs or has_image_files:
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the HAM10000 image dataset directory in the repo or common download locations.")


def resize_image(image: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
    elif rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb = rgb.copy()
    else:
        raise ValueError(f"Unexpected image shape for resize: {rgb.shape}")
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return ensure_uint8_image(resized)


def find_dataset_image(dataset_dir: Path, image_id: str) -> Path | None:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return None
    target = str(image_id).strip().lower()
    for candidate in sorted(dataset_path.rglob("*")):
        if not candidate.is_file():
            continue
        name = candidate.name.lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".bmp")) and target in name:
            return candidate
    return None


def load_original_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return ensure_uint8_image(rgb)


def apply_training_augmentation(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    if np.random.rand() < 0.3:
        image = cv2.flip(image, 0)
    if np.random.rand() < 0.7:
        angle = np.random.uniform(-20.0, 20.0)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        target_h = max(1, int(image.shape[0] * scale))
        target_w = max(1, int(image.shape[1] * scale))
        scaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        orig_h, orig_w = image.shape[:2]

        if target_h >= orig_h and target_w >= orig_w:
            y_start = (target_h - orig_h) // 2
            x_start = (target_w - orig_w) // 2
            image = scaled[y_start:y_start + orig_h, x_start:x_start + orig_w]
        else:
            padded = np.zeros_like(image)
            y_start = (orig_h - target_h) // 2
            x_start = (orig_w - target_w) // 2
            y_end = y_start + target_h
            x_end = x_start + target_w
            padded[y_start:y_end, x_start:x_end] = scaled
            image = padded
    return ensure_uint8_image(image)


def save_processed_image(image_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = ensure_uint8_image(image_rgb)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[-1] != 3:
        raise ValueError(f"Unexpected channel count: {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Processed image must be uint8 before saving but got {image.dtype}.")
    success = cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise IOError(f"Failed to write processed image to {output_path}")


def process_dataset(dataset_dir: Path, split_path: Path, output_dir: Path) -> dict:
    manifest = load_split_manifest(split_path)
    if not check_split_leakage(manifest):
        raise ValueError("Data leakage detected: lesion_id appears in more than one split.")

    processed_dir = Path(output_dir)
    reports_dir = processed_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_count = {split: 0 for split in ALLOWED_SPLITS}
    class_counts = {split: {} for split in ALLOWED_SPLITS}
    corruption_rows = []
    processing_rows = []

    for _, row in manifest.iterrows():
        split_name = str(row["split"]).lower().strip()
        if split_name not in ALLOWED_SPLITS:
            continue

        image_id = str(row["image_id"]).strip()
        lesion_id = str(row["lesion_id"]).strip()
        disease_code = str(row["dx"]).strip()
        image_file = find_dataset_image(dataset_dir, image_id)
        if image_file is None:
            corruption_rows.append({
                "image_id": image_id,
                "lesion_id": lesion_id,
                "dx": disease_code,
                "split": split_name,
                "reason": "image_missing",
            })
            processing_rows.append({
                "image_id": image_id,
                "lesion_id": lesion_id,
                "dx": disease_code,
                "split": split_name,
                "original_shape": None,
                "processed_shape": None,
                "status": "failed",
                "failure_reason": "image_missing",
                "augmentation": "none",
            })
            continue

        try:
            original_rgb = load_original_rgb(image_file)
            original_shape = original_rgb.shape
        except Exception as exc:
            corruption_rows.append({
                "image_id": image_id,
                "lesion_id": lesion_id,
                "dx": disease_code,
                "split": split_name,
                "reason": f"unreadable_image:{type(exc).__name__}",
            })
            processing_rows.append({
                "image_id": image_id,
                "lesion_id": lesion_id,
                "dx": disease_code,
                "split": split_name,
                "original_shape": None,
                "processed_shape": None,
                "status": "failed",
                "failure_reason": f"unreadable_image:{type(exc).__name__}",
                "augmentation": "none",
            })
            continue

        resized_rgb = resize_image(original_rgb, width=TARGET_SIZE[1], height=TARGET_SIZE[0])
        if resized_rgb.shape[:2] != TARGET_SIZE:
            raise ValueError(f"Unexpected resize result shape {resized_rgb.shape} for {image_id}")

        target_dir = processed_dir / split_name / disease_code
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{image_id}.png"

        if split_name == "train":
            processed = apply_training_augmentation(resized_rgb)
            augmentation_label = "training_augmentation"
        else:
            processed = resized_rgb
            augmentation_label = "none"

        save_processed_image(processed, output_path)
        processed_count[split_name] += 1
        class_counts[split_name][disease_code] = class_counts[split_name].get(disease_code, 0) + 1
        processing_rows.append({
            "image_id": image_id,
            "lesion_id": lesion_id,
            "dx": disease_code,
            "split": split_name,
            "original_shape": list(original_shape),
            "processed_shape": list(processed.shape),
            "status": "processed",
            "failure_reason": "",
            "augmentation": augmentation_label,
        })

    stats_rows = []
    for split_name in ALLOWED_SPLITS:
        stats_rows.append({
            "split": split_name,
            "image_count": processed_count[split_name],
            "class_count": len(class_counts[split_name]),
        })

    stats_frame = pd.DataFrame(stats_rows)
    class_frame = pd.DataFrame(
        [
            {"split": split_name, "dx": disease_code, "disease_name": CLASS_NAMES.get(disease_code, "Unknown"), "image_count": count}
            for split_name in ALLOWED_SPLITS
            for disease_code, count in sorted(class_counts[split_name].items())
        ]
    )

    stats_frame.to_csv(reports_dir / "processing_statistics.csv", index=False)
    class_frame.to_csv(reports_dir / "class_wise_statistics.csv", index=False)
    pd.DataFrame(corruption_rows).to_csv(reports_dir / "corrupted_images.csv", index=False)
    pd.DataFrame(processing_rows).to_csv(reports_dir / "processing_report.csv", index=False)
    with open(reports_dir / "preprocessing_summary.json", "w", encoding="utf-8") as fh:
        json.dump(build_summary_json(processed_dir, manifest, len(corruption_rows), processed_count, class_counts), fh, indent=2)
    with open(reports_dir / "processing_summary.json", "w", encoding="utf-8") as fh:
        json.dump(build_summary_json(processed_dir, manifest, len(corruption_rows), processed_count, class_counts), fh, indent=2)

    manifest_dir = processed_dir / "metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_dir / "team1_dataset_split.csv", index=False)
    manifest.to_csv(processed_dir / "team1_dataset_split.csv", index=False)

    if not check_split_leakage(manifest):
        raise ValueError("Split leakage check failed after processing.")

    return {
        "manifest": manifest,
        "split_counts": processed_count,
        "class_counts": class_counts,
        "corrupted_count": len(corruption_rows),
        "stats_frame": stats_frame,
        "class_frame": class_frame,
        "processed_dir": processed_dir,
        "reports_dir": reports_dir,
        "processing_rows": processing_rows,
    }


def generate_before_after_visualization(dataset_dir: Path, manifest_path: Path, output_dir: Path, max_examples: int = 4) -> Path:
    manifest = load_split_manifest(manifest_path)
    sample_rows = manifest.groupby("split", group_keys=False).head(max_examples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_rows.empty:
        save_path = output_dir / "before_after_samples.png"
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No sample images available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return save_path

    fig, axes = plt.subplots(len(sample_rows), 2, figsize=(8, 3 * len(sample_rows)))
    if len(sample_rows) == 1:
        axes = np.array([axes]).reshape(1, 2)

    for idx, row in sample_rows.iterrows():
        image_id = str(row["image_id"])
        source = find_dataset_image(dataset_dir, image_id)
        if source is None:
            continue
        before = load_original_rgb(source)
        after = resize_image(before, 224, 224)
        axes[idx, 0].imshow(before)
        axes[idx, 0].set_title(f"Before: {image_id}\n{row['split']}")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(after)
        axes[idx, 1].set_title(f"After: 224x224 RGB uint8\n{row['split']}")
        axes[idx, 1].axis("off")

    fig.tight_layout()
    save_path = output_dir / "before_after_samples.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def validate_processed_images(base_dir: Path) -> pd.DataFrame:
    records = []
    for image_path in sorted(base_dir.rglob("*.png")):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                records.append({"image_path": str(image_path), "passed": False, "reason": "cv2_imread_failed"})
                continue
            h, w = image.shape[:2]
            valid = image.dtype == np.uint8 and h == 224 and w == 224 and image.min() >= 0 and image.max() <= 255
            records.append({
                "image_path": str(image_path),
                "passed": bool(valid),
                "reason": "ok" if valid else "shape_or_dtype_invalid",
                "height": h,
                "width": w,
                "dtype": str(image.dtype),
            })
        except Exception as exc:
            records.append({"image_path": str(image_path), "passed": False, "reason": f"exception:{type(exc).__name__}"})
    return pd.DataFrame(records)


def build_summary_json(processed_dir: Path, manifest: pd.DataFrame, corrupted_count: int, split_counts: dict[str, int], class_counts: dict[str, dict[str, int]]) -> dict:
    total_input = len(manifest)
    return {
        "total_input_images": total_input,
        "train_count": split_counts.get("train", 0),
        "validation_count": split_counts.get("validation", 0),
        "test_count": split_counts.get("test", 0),
        "successfully_processed_count": sum(split_counts.values()),
        "failed_or_corrupted_count": corrupted_count,
        "image_size": [TARGET_SIZE[0], TARGET_SIZE[1]],
        "color_format": "RGB",
        "normalization_strategy": "Saved as uint8 RGB PNGs in the 0..255 range. Team 3 must not apply a second 1/255 rescale during training.",
        "augmentation_strategy": "Only train data receives random flips, rotation, and zoom. Validation and test are deterministic and unaugmented.",
        "output_location": str(processed_dir),
        "dataset_split_manifest": str(processed_dir / "metadata" / "team1_dataset_split.csv"),
        "class_wise_counts": {split: {key: value for key, value in counts.items()} for split, counts in class_counts.items()},
        "leakage_free": bool(check_split_leakage(manifest)),
    }


def write_team2_handover(processed_dir: Path, split_counts: dict[str, int], corrupted_count: int) -> Path:
    handover = """# Team 2 Dataset Handover

## Processed dataset location
The processed Team 2 dataset is located at: {processed_dir}

## Split locations
- Train: {train_dir}
- Validation: {validation_dir}
- Test: {test_dir}

## Image specification
- Image size: 224 x 224
- Color format: RGB
- Normalization strategy: saved as uint8 RGB PNGs in the 0..255 range; do not apply a second 1/255 rescale in Team 3
- Augmentation: random flips, rotation, and zoom are applied only to training images; validation/test images are deterministic and unaugmented

## Split counts
- Train: {train_count}
- Validation: {validation_count}
- Test: {test_count}
- Failed/corrupted: {corrupted_count}

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
""".format(
        processed_dir=processed_dir,
        train_dir=processed_dir / "train",
        validation_dir=processed_dir / "validation",
        test_dir=processed_dir / "test",
        train_count=split_counts.get("train", 0),
        validation_count=split_counts.get("validation", 0),
        test_count=split_counts.get("test", 0),
        corrupted_count=corrupted_count,
    )
    handover_path = processed_dir / "TEAM2_HANDOVER.md"
    handover_path.write_text(handover, encoding="utf-8")
    return handover_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Team 2 image processing pipeline for HAM10000.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Root directory containing HAM10000 images and image folders.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent.parent / "Team-1" / "TEAM1_OUTPUT" / "team1_dataset_split.csv", help="CSV manifest from Team 1.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "TEAM2_OUTPUT", help="Folder to store processed images and reports.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else find_dataset_dir()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Team 1 manifest not found: {manifest_path}")

    result = process_dataset(dataset_dir, manifest_path, output_dir)
    validation = validate_processed_images(result["processed_dir"])
    summary = build_summary_json(result["processed_dir"], result["manifest"], result["corrupted_count"], result["split_counts"], result["class_counts"])
    (result["reports_dir"] / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (result["reports_dir"] / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation.to_csv(result["reports_dir"] / "processed_images_validation.csv", index=False)
    viz_path = generate_before_after_visualization(dataset_dir, manifest_path, result["processed_dir"] / "visualizations")
    handover_path = write_team2_handover(result["processed_dir"], result["split_counts"], result["corrupted_count"])

    print(f"Dataset root: {dataset_dir}")
    print(f"Processed dataset: {result['processed_dir']}")
    print(f"Leakage check: {'PASSED' if check_split_leakage(result['manifest']) else 'FAILED'}")
    print(f"Corrupted images: {result['corrupted_count']}")
    print(result['split_counts'])
    print(f"Reports: {result['reports_dir']}")
    print(f"Visualization: {viz_path}")
    print(f"Handover: {handover_path}")
    print(f"Processed image validation rows: {len(validation)}; valid rows: {int(validation['passed'].sum())}")


if __name__ == "__main__":
    main()