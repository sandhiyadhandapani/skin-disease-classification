import importlib.util
import json
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


def test_write_class_mapping_creates_json(tmp_path):
    mapping_path = tmp_path / "class_mapping.json"
    MODULE.write_class_mapping(mapping_path)

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping["akiec"] == 0
    assert mapping["vasc"] == 6
    assert list(mapping.keys()) == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_processed_images_remain_uint8_and_not_normalized(tmp_path):
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    saved = MODULE.save_processed_image(image, tmp_path / "output.png")
    assert saved is None
    reloaded = np.asarray(MODULE.cv2.imread(str(tmp_path / "output.png"), MODULE.cv2.IMREAD_COLOR))
    assert reloaded.dtype == np.uint8
    assert reloaded.min() >= 0 and reloaded.max() <= 255
    assert not np.allclose(reloaded.astype(np.float32) / 255.0, np.zeros_like(reloaded, dtype=np.float32))
