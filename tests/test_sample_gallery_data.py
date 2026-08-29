import shutil
from pathlib import Path

import pandas as pd

from scripts.prepare_sample_images import build_sample_manifest, copy_sample_images

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_build_sample_manifest_creates_10_per_class(tmp_path):
    metadata = []
    for disease in CLASS_ORDER:
        for idx in range(12):
            metadata.append({
                "image_id": f"{disease}_{idx:02d}",
                "dx": disease,
                "lesion_id": f"L{disease}_{idx:02d}",
            })
    manifest = pd.DataFrame(metadata)
    selected = build_sample_manifest(manifest, max_per_class=10)

    assert selected["dx"].nunique() == 7
    assert len(selected) == 70
    for disease in CLASS_ORDER:
        assert (selected["dx"] == disease).sum() == 10
        assert selected.loc[selected["dx"] == disease, "image_id"].nunique() == 10


def test_copy_sample_images_creates_70_files(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    metadata = []
    for disease in CLASS_ORDER:
        for idx in range(12):
            image_id = f"{disease}_{idx:02d}"
            metadata.append({"image_id": image_id, "dx": disease, "lesion_id": f"L{disease}_{idx:02d}"})
            target = dataset_dir / f"{image_id}.jpg"
            target.write_bytes(b"fake-image-bytes")
    manifest = pd.DataFrame(metadata)
    output_dir = tmp_path / "sample_images"

    selected = build_sample_manifest(manifest, max_per_class=10)
    copied = copy_sample_images(selected, dataset_dir, output_dir)

    total_files = sum(1 for _ in output_dir.rglob("*.jpg"))
    assert copied == 70
    assert total_files == 70
    assert sorted(p.name for p in output_dir.iterdir() if p.is_dir()) == CLASS_ORDER
    assert all((output_dir / disease).exists() for disease in CLASS_ORDER)
