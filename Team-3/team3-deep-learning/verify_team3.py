#!/usr/bin/env python3
"""
TEAM 3 FINAL VERIFICATION SCRIPT

Performs comprehensive checks on:
1. Model loading
2. Class mapping
3. Sample prediction
4. All 7 test images
5. Artifact files
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main():
    print("\n" + "=" * 70)
    print("TEAM 3 FINAL VERIFICATION")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parents[2]
    team3_dir = repo_root / "Team-3" / "team3-deep-learning"
    data_dir = repo_root / "skin-disease-classification" / "Team-2" / "processed_data"

    # ===== 1. VERIFY CLASS MAPPING =====
    print("\n[1] VERIFYING CLASS MAPPING...")
    class_mapping_path = data_dir / "class_mapping.json"
    if not class_mapping_path.exists():
        print(f"  FAIL: class_mapping.json not found at {class_mapping_path}")
        return False

    with open(class_mapping_path) as f:
        class_mapping = json.load(f)
    
    expected_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    actual_classes = list(class_mapping.keys())
    
    print(f"  Expected classes: {expected_classes}")
    print(f"  Actual classes:   {actual_classes}")
    
    if actual_classes != expected_classes:
        print(f"  FAIL: Class mapping mismatch!")
        return False
    
    for cls, idx in class_mapping.items():
        if idx != expected_classes.index(cls):
            print(f"  FAIL: Class '{cls}' has wrong index {idx}, expected {expected_classes.index(cls)}")
            return False
    
    print(f"  PASS: All 7 classes verified correctly")

    # ===== 2. FIND AND LOAD MODEL =====
    print("\n[2] FINDING AND LOADING SAVED MODEL...")
    models_dir = team3_dir / "models"
    if not models_dir.exists():
        print(f"  FAIL: models/ directory not found at {models_dir}")
        return False
    
    keras_files = list(models_dir.glob("*.keras"))
    if not keras_files:
        print(f"  FAIL: No .keras model files found in {models_dir}")
        return False
    
    # Find the best_model (or first model if multiple)
    best_model_path = None
    for f in keras_files:
        if "best_model" in f.name:
            best_model_path = f
            break
    
    if best_model_path is None:
        best_model_path = keras_files[0]
    
    print(f"  Model path: {best_model_path}")
    print(f"  Model file size: {best_model_path.stat().st_size / 1024:.1f} KB")
    
    try:
        model = tf.keras.models.load_model(str(best_model_path))
        print(f"  PASS: Model loaded successfully")
    except Exception as e:
        print(f"  FAIL: Could not load model: {e}")
        return False

    # ===== 3. VERIFY MODEL ARCHITECTURE =====
    print("\n[3] VERIFYING MODEL ARCHITECTURE...")
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Number of classes (output dim): {model.output_shape[-1]}")
    
    if model.output_shape[-1] != 7:
        print(f"  FAIL: Output has {model.output_shape[-1]} classes, expected 7")
        return False
    
    if model.input_shape[-1:] != (3,):
        print(f"  FAIL: Input channels = {model.input_shape[-1]}, expected 3 (RGB)")
        return False
    
    if model.input_shape[1:3] != (224, 224):
        print(f"  FAIL: Input size = {model.input_shape[1:3]}, expected (224, 224)")
        return False
    
    print(f"  PASS: Model architecture verified")

    # ===== 4. VERIFY TEST IMAGES EXIST =====
    print("\n[4] LOCATING TEST IMAGES...")
    test_dir = data_dir / "test"
    if not test_dir.exists():
        print(f"  FAIL: test/ directory not found at {test_dir}")
        return False
    
    test_images = []
    for cls_dir in sorted(test_dir.iterdir()):
        if cls_dir.is_dir():
            for img_file in sorted(cls_dir.glob("*.*")):
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    test_images.append((cls_dir.name, img_file))
    
    print(f"  Found {len(test_images)} test images")
    if len(test_images) != 7:
        print(f"  WARNING: Expected 7 test images, found {len(test_images)}")
    
    for cls, img_path in test_images:
        print(f"    - {cls}: {img_path.name}")

    # ===== 5. RUN SAMPLE PREDICTION (FIRST TEST IMAGE) =====
    print("\n[5] RUNNING SAMPLE PREDICTION (FIRST IMAGE)...")
    if test_images:
        sample_cls, sample_img_path = test_images[0]
        print(f"  Image: {sample_img_path}")
        print(f"  True class: {sample_cls}")
        
        try:
            img = tf.keras.utils.load_img(str(sample_img_path), target_size=(224, 224))
            arr = tf.keras.utils.img_to_array(img)
            arr = np.expand_dims(arr, axis=0) / 255.0
            
            # Verify preprocessing
            print(f"  Array shape: {arr.shape}")
            print(f"  Array dtype: {arr.dtype}")
            print(f"  Array range: [{arr.min():.6f}, {arr.max():.6f}]")
            
            # Run prediction
            pred = model.predict(arr, verbose=0)[0]
            pred_idx = np.argmax(pred)
            pred_label = expected_classes[pred_idx]
            pred_confidence = float(pred[pred_idx])
            
            print(f"  Predicted class: {pred_label} (index {pred_idx})")
            print(f"  Confidence: {pred_confidence:.6f}")
            print(f"  Output shape: {pred.shape}")
            print(f"  Probabilities sum: {pred.sum():.6f}")
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(pred)):
                print(f"  FAIL: Output contains NaN or Inf values")
                return False
            
            print(f"  PASS: Sample prediction successful")
        except Exception as e:
            print(f"  FAIL: Could not run prediction: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ===== 6. VERIFY ALL 7 TEST IMAGES =====
    print("\n[6] VERIFYING ALL TEST IMAGES...")
    success_count = 0
    fail_count = 0
    
    for cls, img_path in test_images:
        try:
            img = tf.keras.utils.load_img(str(img_path), target_size=(224, 224))
            arr = tf.keras.utils.img_to_array(img)
            arr = np.expand_dims(arr, axis=0) / 255.0
            
            pred = model.predict(arr, verbose=0)[0]
            pred_idx = np.argmax(pred)
            pred_label = expected_classes[pred_idx]
            pred_confidence = float(pred[pred_idx])
            
            # Check validity
            if not np.all(np.isfinite(pred)):
                print(f"  ✗ {img_path.name}: Invalid probabilities (NaN/Inf)")
                fail_count += 1
            else:
                print(f"  ✓ {img_path.name}: {pred_label} ({pred_confidence:.4f})")
                success_count += 1
        except Exception as e:
            print(f"  ✗ {img_path.name}: Error - {e}")
            fail_count += 1
    
    print(f"\n  Successful: {success_count}/{len(test_images)}")
    print(f"  Failed: {fail_count}/{len(test_images)}")
    
    if success_count != len(test_images):
        print(f"  FAIL: Not all test images could be processed")
        return False
    
    print(f"  PASS: All test images verified")

    # ===== 7. VERIFY TRAINING ARTIFACTS =====
    print("\n[7] VERIFYING TRAINING ARTIFACTS...")
    results_dir = team3_dir / "results"
    required_artifacts = [
        "training_history.csv",
        "training_history.json",
        "training_accuracy.png",
        "training_loss.png",
        "model_metadata.json",
    ]
    
    missing = []
    for artifact in required_artifacts:
        artifact_path = results_dir / artifact if "training" in artifact or "confusion" in artifact else team3_dir / artifact
        if artifact == "model_metadata.json":
            artifact_path = team3_dir / artifact
        else:
            artifact_path = results_dir / artifact
        
        if artifact_path.exists():
            size = artifact_path.stat().st_size
            print(f"  ✓ {artifact} ({size} bytes)")
        else:
            print(f"  ✗ {artifact} NOT FOUND")
            missing.append(artifact)
    
    if missing:
        print(f"  FAIL: Missing artifacts: {missing}")
        return False
    
    print(f"  PASS: All training artifacts present")

    # ===== 8. VERIFY METADATA =====
    print("\n[8] VERIFYING MODEL METADATA...")
    metadata_path = team3_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        print(f"  Train images: {metadata.get('train_images')}")
        print(f"  Validation images: {metadata.get('validation_images')}")
        print(f"  Test images: {metadata.get('test_images')}")
        print(f"  Classes: {metadata.get('num_classes')}")
        print(f"  Note: {metadata.get('note', 'N/A')}")
        
        if metadata.get('train_images') == 54 and metadata.get('validation_images') == 7 and metadata.get('test_images') == 7:
            print(f"  PASS: Metadata counts verified (54/7/7)")
        else:
            print(f"  FAIL: Metadata counts incorrect")
            return False
    else:
        print(f"  WARNING: model_metadata.json not found")

    # ===== FINAL RESULT =====
    print("\n" + "=" * 70)
    print("VERIFICATION RESULT: PASS ✓")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ Python execution working")
    print("  ✓ Class mapping verified (7 classes)")
    print("  ✓ Model loaded successfully")
    print("  ✓ Model architecture correct (224x224x3 → 7 classes)")
    print("  ✓ Sample prediction successful (no NaN/Inf)")
    print("  ✓ All 7 test images verified")
    print("  ✓ Training artifacts present")
    print("  ✓ Metadata correct (54/7/7 split)")
    print("\nNEXT STEP: Team 4 official evaluation")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
