import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "skin-disease-classification" / "Team-2" / "team2_image_processing.py"
SPEC = importlib.util.spec_from_file_location("team2_image_processing", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

check_split_leakage = MODULE.check_split_leakage
ensure_uint8_image = MODULE.ensure_uint8_image
load_split_manifest = MODULE.load_split_manifest
resize_image = MODULE.resize_image


def test_check_split_leakage_detects_overlap():
    split = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "lesion_id": ["L1", "L2", "L1"],
            "dx": ["nv", "mel", "nv"],
            "split": ["train", "validation", "test"],
        }
    )
    assert check_split_leakage(split) is False


def test_ensure_uint8_image_converts_to_uint8():
    image = np.array([[[0.0, 0.5, 1.0], [255.0, 255.0, 255.0]]], dtype=np.float32)
    updated = ensure_uint8_image(image)
    assert updated.dtype == np.uint8
    assert updated.min() == 0
    assert updated.max() <= 255


def test_resize_image_224():
    arr = np.zeros((64, 128, 3), dtype=np.uint8)
    out = resize_image(arr, 224, 224)
    assert out.shape == (224, 224, 3)


def test_load_split_manifest_uses_team1_file(tmp_path):
    manifest = tmp_path / "team1_dataset_split.csv"
    pd.DataFrame(
        {
            "image_id": ["img1", "img2"],
            "lesion_id": ["L1", "L2"],
            "dx": ["nv", "mel"],
            "split": ["train", "test"],
        }
    ).to_csv(manifest, index=False)

    loaded = load_split_manifest(manifest)
    assert list(loaded.columns) == ["image_id", "lesion_id", "dx", "split"]
    assert set(loaded["split"].unique()) == {"train", "test"}
