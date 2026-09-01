from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_SIZE = (224, 224)
ALLOWED_SPLITS = ["train", "validation", "test"]
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def write_class_mapping(output_path: str | Path) -> dict[str, int]:
    path = Path(output_path)
    mapping = {class_name: idx for idx, class_name in enumerate(CLASS_ORDER)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


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
    split_df["dx"] = split_df["dx"].astype(str).str.lower().str.strip()
    return split_df


def check_split_leakage(split_df: pd.DataFrame) -> bool:
    split_sets = {
        split_name: set(split_df.loc[split_df["split"] == split_name, "lesion_id"])
        for split_name in ALLOWED_SPLITS
    }
    for left_idx, left_split in enumerate(ALLOWED_SPLITS):
        for right_split in ALLOWED_SPLITS[left_idx + 1 :]:
            if split_sets[left_split] & split_sets[right_split]:
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
        Path(__file__).resolve().parent.parent / "sample_images",
    ]
    for candidate in candidates:
        if candidate.exists() and ((candidate / "HAM10000_metadata.csv").exists() or (candidate / "sample_images").exists()):
            return candidate.resolve()
    raise FileNotFoundError("Could not locate HAM10000_metadata.csv or the project sample gallery.")


def normalize_image_id(path: Path) -> str:
    name = path.name
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        lower = name.lower()
        if lower.endswith(suffix + suffix):
            name = name[: -len(suffix)]
        if lower.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def resolve_image_candidates(dataset_dir: Path, image_id: str, disease_code: str | None = None) -> list[Path]:
    image_dir = Path(dataset_dir)
    if not image_dir.exists():
        return []
    candidates: list[Path] = []
    normalized_target = image_id.strip().lower()
    for current in [image_dir, image_dir / "sample_images"]:
        for path in current.rglob("*"):
            if not path.is_file():
                continue
            current_name = normalize_image_id(path).lower()
            if current_name == normalized_target or path.stem.lower() == normalized_target or path.name.lower() == normalized_target:
                candidates.append(path)
            if disease_code and path.parent.name.lower() == disease_code.lower() and current_name == normalized_target:
                candidates.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def find_dataset_image(dataset_dir: Path, image_id: str, disease_code: str | None = None) -> Path | None:
    matches = resolve_image_candidates(dataset_dir, image_id, disease_code)
    if not matches:
        return None
    return matches[0]


def load_original_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[-1] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported image channel count in {image_path}: {image.shape}")
    return ensure_uint8_image(rgb)


def resize_image(image: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    if rgb.shape[-1] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
    if rgb.shape[-1] == 3 and rgb.dtype != np.uint8:
        rgb = ensure_uint8_image(rgb)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return ensure_uint8_image(resized)


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
        new_h = max(1, int(image.shape[0] * scale))
        new_w = max(1, int(image.shape[1] * scale))
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        y_offset = (image.shape[0] - new_h) // 2
        x_offset = (image.shape[1] - new_w) // 2
        padded = np.zeros_like(image)

        y_start = max(0, y_offset)
        x_start = max(0, x_offset)
        y_end = min(image.shape[0], y_start + new_h)
        x_end = min(image.shape[1], x_start + new_w)
        src_y_start = max(0, -y_offset)
        src_x_start = max(0, -x_offset)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_end = src_x_start + (x_end - x_start)

        padded[y_start:y_end, x_start:x_end] = scaled[src_y_start:src_y_end, src_x_start:src_x_end]
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

    success = cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise IOError(f"Failed to write processed image to {output_path}")


def build_sample_manifest(dataset_dir: Path) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    records = []
    disease_name_map = {
        "akiec_actinic_keratoses": "akiec",
        "bcc_basal_cell_carcinoma": "bcc",
        "bkl_benign_keratosis": "bkl",
        "df_dermatofibroma": "df",
        "mel_melanoma": "mel",
        "nv_melanocytic_nevi": "nv",
        "vasc_vascular_lesion": "vasc",
    }

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        disease_code = disease_name_map.get(class_dir.name, class_dir.name)
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_id = normalize_image_id(file_path)
                records.append({
                    "image_id": image_id,
                    "lesion_id": f"sample_{disease_code}_{image_id}",
                    "dx": disease_code,
                    "split": "train",
                })

    if not records:
        raise FileNotFoundError(f"No sample images found under {dataset_dir}")

    frame = pd.DataFrame(records)
    results = {split: [] for split in ALLOWED_SPLITS}
    for disease_code in CLASS_ORDER:
        disease_rows = frame[frame["dx"] == disease_code].reset_index(drop=True)
        if disease_rows.empty:
            continue
        shuffled = disease_rows.sample(frac=1, random_state=42).reset_index(drop=True)
        total = len(shuffled)
        train_n = max(1, int(round(total * 0.8))) if total > 1 else total
        val_n = max(1, int(round(total * 0.1))) if total > 2 else 0
        test_n = max(0, total - train_n - val_n)
        if total == 1:
            cuts = {"train": 1, "validation": 1, "test": 1}
        else:
            cuts = {"train": train_n, "validation": train_n + val_n, "test": total}
        for split_name in ALLOWED_SPLITS:
            start = 0 if split_name == "train" else cuts[ALLOWED_SPLITS[ALLOWED_SPLITS.index(split_name) - 1]]
            end = cuts[split_name]
            if end > start:
                results[split_name].append(shuffled.iloc[start:end].assign(split=split_name))

    concatenated = [pd.concat(values, ignore_index=True) for values in results.values() if values]
    result = pd.concat(concatenated, ignore_index=True) if concatenated else pd.DataFrame(columns=["image_id", "lesion_id", "dx", "split"])
    return result[["image_id", "lesion_id", "dx", "split"]]


def process_dataset(dataset_dir: Path, split_path: Path, output_dir: Path) -> dict:
    manifest_path = Path(split_path)
    dataset_root = Path(dataset_dir)
    if not manifest_path.exists():
        if dataset_root.is_dir() and any(child.is_dir() for child in dataset_root.iterdir()):
            manifest = build_sample_manifest(dataset_root)
        elif (dataset_root / "sample_images").exists():
            manifest = build_sample_manifest(dataset_root / "sample_images")
        else:
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    else:
        manifest = load_split_manifest(manifest_path)

    if not check_split_leakage(manifest):
        raise ValueError("Data leakage detected: lesion_id appears in more than one split.")

    processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    class_counts = {split: {} for split in ALLOWED_SPLITS}
    split_counts = {split: 0 for split in ALLOWED_SPLITS}
    corrupted_rows = []

    for _, row in manifest.iterrows():
        split_name = str(row["split"]).lower().strip()
        if split_name not in ALLOWED_SPLITS:
            continue

        image_id = str(row["image_id"]).strip()
        disease_code = str(row["dx"]).strip().lower()
        source = find_dataset_image(Path(dataset_dir), image_id, disease_code)
        if source is None:
            corrupted_rows.append({"image_id": image_id, "lesion_id": row["lesion_id"], "dx": disease_code, "split": split_name, "reason": "image_missing"})
            continue

        try:
            original_rgb = load_original_rgb(source)
        except Exception:
            corrupted_rows.append({"image_id": image_id, "lesion_id": row["lesion_id"], "dx": disease_code, "split": split_name, "reason": "unreadable_image"})
            continue

        processed_rgb = resize_image(original_rgb, width=TARGET_SIZE[1], height=TARGET_SIZE[0])
        if processed_rgb.shape[:2] != TARGET_SIZE:
            raise ValueError(f"Unexpected resize result shape {processed_rgb.shape} for {image_id}")

        dest_dir = processed_dir / split_name / disease_code
        dest_dir.mkdir(parents=True, exist_ok=True)
        output_path = dest_dir / f"{image_id}.png"
        if split_name == "train":
            processed_rgb = apply_training_augmentation(processed_rgb)
        save_processed_image(processed_rgb, output_path)

        split_counts[split_name] += 1
        class_counts[split_name][disease_code] = class_counts[split_name].get(disease_code, 0) + 1

    class_mapping_path = processed_dir / "class_mapping.json"
    write_class_mapping(class_mapping_path)

    stats_dir = processed_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"split": split_name, "image_count": split_counts[split_name], "class_count": len(class_counts[split_name])}
        for split_name in ALLOWED_SPLITS
    ]).to_csv(stats_dir / "processing_statistics.csv", index=False)

    class_summary_rows = []
    for split_name in ALLOWED_SPLITS:
        for disease_code, count in sorted(class_counts[split_name].items()):
            class_summary_rows.append({
                "split": split_name,
                "dx": disease_code,
                "disease_name": CLASS_NAMES.get(disease_code, "Unknown"),
                "image_count": count,
            })
    pd.DataFrame(class_summary_rows).to_csv(stats_dir / "class_wise_statistics.csv", index=False)
    pd.DataFrame(corrupted_rows).to_csv(stats_dir / "corrupted_images.csv", index=False)
    if not corrupted_rows:
        pd.DataFrame(columns=["image_id", "lesion_id", "dx", "split", "reason"]).to_csv(stats_dir / "corrupted_images.csv", index=False)

    metadata_dir = processed_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(metadata_dir / "team1_dataset_split.csv", index=False)

    summary = {
        "dataset_split_manifest": str(metadata_dir / "team1_dataset_split.csv"),
        "class_mapping_path": str(class_mapping_path),
        "class_wise_counts": {split: counts for split, counts in class_counts.items()},
        "split_counts": split_counts,
        "corrupted_image_count": len(corrupted_rows),
        "leakage_free": bool(check_split_leakage(manifest)),
        "processed_dataset_root": str(processed_dir),
        "normalization_strategy": "RGB uint8 arrays in 0..255 saved to disk; normalization performed exactly once in the model loader",
    }
    (processed_dir / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    validation = validate_processed_images(processed_dir)
    validation.to_csv(processed_dir / "processed_images_validation.csv", index=False)

    return {
        "manifest": manifest,
        "processed_dir": processed_dir,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "corrupted_count": len(corrupted_rows),
        "validation": validation,
    }


def generate_before_after_visualization(dataset_dir: Path, manifest_or_path: Path | pd.DataFrame, output_dir: Path, max_examples: int = 4) -> Path:
    if isinstance(manifest_or_path, pd.DataFrame):
        manifest = manifest_or_path.copy()
    else:
        manifest = load_split_manifest(manifest_or_path)
    sample_rows = manifest.groupby("split", group_keys=False).head(max_examples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for _, row in sample_rows.iterrows():
        image_id = str(row["image_id"])
        source = find_dataset_image(Path(dataset_dir), image_id, str(row["dx"]).strip().lower())
        if source is None:
            continue
        raw = load_original_rgb(source)
        processed = resize_image(raw, 224, 224)
        examples.append((image_id, row["split"], raw, processed))

    if not examples:
        placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
        examples.append(("placeholder", "train", placeholder.copy(), placeholder.copy()))

    before_original_path = output_dir / "before_original.png"
    cv2.imwrite(str(before_original_path), cv2.cvtColor(examples[0][2], cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(len(examples), 2, figsize=(8, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.array([axes]).reshape(1, 2)

    for idx, (image_id, split_name, before, after) in enumerate(examples):
        axes[idx, 0].imshow(before)
        axes[idx, 0].set_title(f"Original raw image\n{image_id}\n{split_name}")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(after)
        axes[idx, 1].set_title(f"Processed 224x224 RGB uint8\n{image_id}\n{split_name}")
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
            height, width = image.shape[:2]
            valid = image.dtype == np.uint8 and height == 224 and width == 224 and image.min() >= 0 and image.max() <= 255
            records.append({
                "image_path": str(image_path),
                "passed": bool(valid),
                "reason": "ok" if valid else "shape_or_dtype_invalid",
                "height": height,
                "width": width,
                "dtype": str(image.dtype),
            })
        except Exception as exc:
            records.append({"image_path": str(image_path), "passed": False, "reason": f"exception:{type(exc).__name__}"})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Team 2 image processing pipeline for HAM10000.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Root directory containing either HAM10000_metadata.csv or a class-based sample dataset.")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent.parent / "Team-1" / "TEAM1_OUTPUT" / "team1_dataset_split.csv", help="CSV manifest from Team 1.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "processed_data", help="Folder to store processed images and metadata.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else find_dataset_dir()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_dir}")

    result = process_dataset(dataset_dir, manifest_path, output_dir)
    validation = result["validation"]
    print(f"Processed dataset: {result['processed_dir']}")
    print(f"Leakage check: {'PASSED' if check_split_leakage(result['manifest']) else 'FAILED'}")
    print(f"Corrupted images: {result['corrupted_count']}")
    print(result['split_counts'])
    print(f"Processed image validation rows: {len(validation)}; valid rows: {int(validation['passed'].sum())}")
    viz_path = generate_before_after_visualization(dataset_dir, result["manifest"], result["processed_dir"] / "visualizations")
    print(f"Visualization: {viz_path}")


if __name__ == "__main__":
    main()
