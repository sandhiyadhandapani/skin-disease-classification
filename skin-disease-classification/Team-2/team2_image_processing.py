from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
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
        split_name: set(split_df.loc[split_df["split"] == split_name, "lesion_id"])
        for split_name in ALLOWED_SPLITS
    }
    overlaps = []
    for left_idx, left_split in enumerate(ALLOWED_SPLITS):
        for right_split in ALLOWED_SPLITS[left_idx + 1 :]:
            overlap = split_sets[left_split] & split_sets[right_split]
            if overlap:
                overlaps.append({"left_split": left_split, "right_split": right_split, "lesion_ids": sorted(overlap)})
    if overlaps:
        return False
    return True


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image array is None.")
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.copy()
    if image.dtype.kind in {"f", "i"}:
        image = np.clip(image, 0, 255)
        if image.dtype.kind == "f":
            image = np.rint(image).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        return image
    return image.astype(np.uint8)


def find_dataset_dir() -> Path:
    candidates = [
        Path.home() / "Downloads" / "archive (1)",
        Path.home() / "Downloads" / "HAM10000",
        Path.home() / "Downloads",
        Path(__file__).resolve().parents[2],
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "HAM10000_metadata.csv").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate HAM10000_metadata.csv in common Downloads or project directories.")


def resize_image(image: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    if rgb.shape[-1] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return ensure_uint8_image(resized)


def find_dataset_image(dataset_dir: Path, image_id: str) -> Path | None:
    image_dir = Path(dataset_dir)
    if not image_dir.exists():
        return None
    for candidate in image_dir.rglob("*"):
        if candidate.is_file() and candidate.stem == image_id:
            return candidate
    return None


def load_original_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return ensure_uint8_image(rgb)


def apply_training_augmentation(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.copy().astype(np.uint8)
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
        new_size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
        scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        dx = int((new_size[0] - image.shape[1]) / 2)
        dy = int((new_size[1] - image.shape[0]) / 2)
        padded = np.zeros_like(image)
        x_start = max(0, dx)
        y_start = max(0, dy)
        x_end = min(image.shape[1], x_start + new_size[0])
        y_end = min(image.shape[0], y_start + new_size[1])
        src_x = max(0, -dx)
        src_y = max(0, -dy)
        padded[y_start:y_end, x_start:x_end] = scaled[src_y:src_y + (y_end - y_start), src_x:src_x + (x_end - x_start)]
        image = padded
    return ensure_uint8_image(image)


def save_processed_image(image_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = ensure_uint8_image(image_rgb)
    if image.dtype != np.uint8:
        raise ValueError(f"Processed image must be uint8 before saving but got {image.dtype}.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 3:
        pass
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unexpected channel count: {image.shape}")
    image_pil = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(output_path), image_pil)
    if not success:
        raise IOError(f"Failed to write processed image to {output_path}")


def process_dataset(dataset_dir: Path, split_path: Path, output_dir: Path) -> dict:
    manifest = load_split_manifest(split_path)
    leakage_ok = check_split_leakage(manifest)
    if not leakage_ok:
        raise ValueError("Data leakage detected: lesion_id appears in more than one split.")

    processed_dir = Path(output_dir)
    stats_rows = []
    corrupted_rows = []
    processed_count = {split: 0 for split in ALLOWED_SPLITS}
    class_counts = {split: {} for split in ALLOWED_SPLITS}

    for _, row in manifest.iterrows():
        split_name = str(row["split"]).lower().strip()
        if split_name not in ALLOWED_SPLITS:
            continue

        image_id = str(row["image_id"]).strip()
        disease_code = str(row["dx"]).strip()
        image_file = find_dataset_image(dataset_dir, image_id)
        if image_file is None:
            corrupted_rows.append({"image_id": image_id, "lesion_id": row["lesion_id"], "dx": disease_code, "split": split_name, "reason": "image_missing"})
            continue

        try:
            original_rgb = load_original_rgb(image_file)
        except Exception:
            corrupted_rows.append({"image_id": image_id, "lesion_id": row["lesion_id"], "dx": disease_code, "split": split_name, "reason": "unreadable_image"})
            continue

        resized_rgb = resize_image(original_rgb, width=TARGET_SIZE[1], height=TARGET_SIZE[0])
        if resized_rgb.shape[:2] != TARGET_SIZE:
            raise ValueError(f"Unexpected resize result shape {resized_rgb.shape} for {image_id}")

        dest_dir = processed_dir / split_name / disease_code
        dest_dir.mkdir(parents=True, exist_ok=True)
        output_path = dest_dir / f"{image_id}.png"

        if split_name == "train":
            augmented = apply_training_augmentation(resized_rgb)
            save_processed_image(augmented, output_path)
            processed_count[split_name] += 1
            class_counts[split_name][disease_code] = class_counts[split_name].get(disease_code, 0) + 1
        else:
            save_processed_image(resized_rgb, output_path)
            processed_count[split_name] += 1
            class_counts[split_name][disease_code] = class_counts[split_name].get(disease_code, 0) + 1

    for split_name in ALLOWED_SPLITS:
        stats_rows.append({
            "split": split_name,
            "image_count": processed_count[split_name],
            "class_count": len(class_counts[split_name]),
        })

    stats_frame = pd.DataFrame(stats_rows)
    class_stats = []
    for split_name in ALLOWED_SPLITS:
        for disease_code, count in sorted(class_counts[split_name].items()):
            class_stats.append({"split": split_name, "dx": disease_code, "disease_name": CLASS_NAMES.get(disease_code, "Unknown"), "image_count": count})
    class_frame = pd.DataFrame(class_stats)

    stats_dir = processed_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_frame.to_csv(stats_dir / "processing_statistics.csv", index=False)
    class_frame.to_csv(stats_dir / "class_wise_statistics.csv", index=False)

    if corrupted_rows:
        pd.DataFrame(corrupted_rows).to_csv(stats_dir / "corrupted_images.csv", index=False)
    else:
        pd.DataFrame(columns=["image_id", "lesion_id", "dx", "split", "reason"]).to_csv(stats_dir / "corrupted_images.csv", index=False)

    manifest_file = processed_dir / "metadata" / "team1_dataset_split.csv"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_file, index=False)

    if not check_split_leakage(manifest):
        raise ValueError("Split leakage check failed after processing.")

    return {
        "manifest": manifest,
        "split_counts": processed_count,
        "class_counts": class_counts,
        "corrupted_count": len(corrupted_rows),
        "stats_frame": stats_frame,
        "class_frame": class_frame,
        "processed_dir": processed_dir,
    }


def generate_before_after_visualization(dataset_dir: Path, manifest_path: Path, output_dir: Path, max_examples: int = 4) -> Path:
    manifest = load_split_manifest(manifest_path)
    sample_rows = manifest.groupby("split", group_keys=False).head(max_examples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        ax_before = axes[idx, 0]
        ax_after = axes[idx, 1]
        ax_before.imshow(before)
        ax_before.set_title(f"Before: {image_id}\n{row['split']}")
        ax_before.axis("off")
        ax_after.imshow(after)
        ax_after.set_title(f"After: 224x224 RGB uint8\n{row['split']}")
        ax_after.axis("off")

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
            records.append({"image_path": str(image_path), "passed": bool(valid), "reason": "ok" if valid else "shape_or_dtype_invalid", "height": h, "width": w, "dtype": str(image.dtype)})
        except Exception as exc:  # pragma: no cover - defensive check
            records.append({"image_path": str(image_path), "passed": False, "reason": f"exception:{type(exc).__name__}"})
    return pd.DataFrame(records)


def build_summary_json(processed_dir: Path, manifest: pd.DataFrame, corrupted_count: int, split_counts: dict[str, int], class_counts: dict[str, dict[str, int]]) -> dict:
    summary = {
        "dataset_split_manifest": str(processed_dir / "metadata" / "team1_dataset_split.csv"),
        "train_image_count": split_counts.get("train", 0),
        "validation_image_count": split_counts.get("validation", 0),
        "test_image_count": split_counts.get("test", 0),
        "corrupted_image_count": corrupted_count,
        "class_wise_counts": {split: {key: value for key, value in counts.items()} for split, counts in class_counts.items()},
        "leakage_free": bool(check_split_leakage(manifest)),
        "processed_dataset_root": str(processed_dir),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Team 2 image processing pipeline for HAM10000.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Root directory containing HAM10000_metadata.csv and image files.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent.parent / "Team-1" / "TEAM1_OUTPUT" / "team1_dataset_split.csv", help="CSV manifest from Team 1.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "TEAM2_OUTPUT", help="Folder to store processed images and statistics.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else find_dataset_dir()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_dir}")
    if not (dataset_dir / "HAM10000_metadata.csv").exists():
        raise FileNotFoundError(f"HAM10000_metadata.csv not found in dataset root: {dataset_dir}")

    result = process_dataset(dataset_dir, manifest_path, output_dir)
    validation = validate_processed_images(result["processed_dir"])
    summary = build_summary_json(result["processed_dir"], result["manifest"], result["corrupted_count"], result["split_counts"], result["class_counts"])
    (output_dir / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "processed_images_validation.csv").to_csv if False else None
    validation.to_csv(output_dir / "processed_images_validation.csv", index=False)
    viz_path = generate_before_after_visualization(dataset_dir, manifest_path, output_dir / "visualizations")

    print(f"Processed dataset: {result['processed_dir']}")
    print(f"Leakage check: {'PASSED' if check_split_leakage(result['manifest']) else 'FAILED'}")
    print(f"Corrupted images: {result['corrupted_count']}")
    print(result['split_counts'])
    print(f"Visualization: {viz_path}")
    print(f"Processed image validation rows: {len(validation)}; valid rows: {int(validation['passed'].sum())}")


if __name__ == "__main__":
    main()
