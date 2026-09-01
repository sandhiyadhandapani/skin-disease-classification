"""Team 3 training pipeline using the repository's available Team 2 output."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import CLASS_NAMES, IMG_SIZE, NUM_CLASSES, build_cnn

AUGMENT_IN_TEAM3 = True


def load_class_mapping(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    ordered = [key for key in mapping.keys()]
    return mapping, ordered


def get_data_generators(data_dir, batch_size, class_names=None):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    test_dir = data_dir / "test"

    for d in (train_dir, val_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Expected Team 2 output folder not found: {d}")

    if class_names is None:
        class_names = CLASS_NAMES

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
    ) if AUGMENT_IN_TEAM3 else ImageDataGenerator(rescale=1.0 / 255)

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=class_names,
        shuffle=True,
        seed=42,
    )
    val_gen = val_test_datagen.flow_from_directory(
        str(val_dir),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=class_names,
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        str(test_dir),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=class_names,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def compute_class_weights(train_gen):
    labels = train_gen.classes
    classes = np.unique(labels)
    counts = np.bincount(labels, minlength=len(classes))
    weights = np.sum(counts) / (len(classes) * counts.astype(np.float32))
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="train_accuracy")
    axes[0].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    path = Path(out_dir) / "training_history.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved training curves to {path}")

    # Also save separate accuracy and loss plots for the Team 3 handover.
    acc_fig, acc_ax = plt.subplots(figsize=(8, 5))
    acc_ax.plot(history.history["accuracy"], label="train_accuracy")
    acc_ax.plot(history.history["val_accuracy"], label="val_accuracy")
    acc_ax.set_title("Training and Validation Accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend()
    acc_fig.tight_layout()
    acc_fig.savefig(Path(out_dir) / "training_accuracy.png")
    plt.close(acc_fig)

    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
    loss_ax.plot(history.history["loss"], label="train_loss")
    loss_ax.plot(history.history["val_loss"], label="val_loss")
    loss_ax.set_title("Training and Validation Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()
    loss_fig.tight_layout()
    loss_fig.savefig(Path(out_dir) / "training_loss.png")
    plt.close(loss_fig)


def evaluate_on_test(model, test_gen, out_dir):
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )
    print("\n=== Test set classification report ===")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred)

    with open(Path(out_dir) / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    np.savetxt(Path(out_dir) / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Test Set")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confusion_matrix.png")
    plt.close(fig)

    return report


def save_training_history(history, out_dir):
    out_path = Path(out_dir)
    metrics = {}
    for key, values in history.history.items():
        metrics[key] = [float(v) for v in values]

    (out_path / "training_history.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    records = []
    for epoch in range(len(next(iter(metrics.values())))):
        row = {"epoch": epoch}
        for key, values in metrics.items():
            row[key] = values[epoch]
        records.append(row)
    import pandas as pd
    pd.DataFrame(records).to_csv(out_path / "training_history.csv", index=False)


def verify_model(model_path, class_names, data_dir):
    model = tf.keras.models.load_model(model_path)
    print(f"Reloaded model: {model_path}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    test_dir = Path(data_dir) / "test"
    all_files = []
    for cls in sorted(p.name for p in test_dir.iterdir() if p.is_dir()):
        all_files.extend(sorted((test_dir / cls).glob("*.*")))

    if not all_files:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    sample_img = tf.keras.utils.load_img(str(all_files[0]), target_size=IMG_SIZE)
    sample_arr = tf.keras.utils.img_to_array(sample_img)
    sample_arr = np.expand_dims(sample_arr, axis=0) / 255.0
    probs = model.predict(sample_arr, verbose=0)[0]
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model output contains NaN or Inf values.")
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    pred_conf = float(probs[pred_idx])
    print(f"Sample prediction: {pred_label} @ {pred_conf:.4f}")

    test_predictions = []
    for file_path in all_files:
        img = tf.keras.utils.load_img(str(file_path), target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0) / 255.0
        pred = model.predict(arr, verbose=0)[0]
        if not np.all(np.isfinite(pred)):
            raise ValueError(f"Prediction for {file_path} contains NaN/Inf.")
        idx = int(np.argmax(pred))
        test_predictions.append({"file": str(file_path), "predicted_index": idx, "predicted_label": class_names[idx], "confidence": float(pred[idx])})
    return model, test_predictions


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_data_dir = repo_root / "Team-2" / "processed_data"

    parser = argparse.ArgumentParser(description="Train Team 3 CNN on the repository's available Team 2 processed data")
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir), help="Path to Team 2 processed data (train/validation/test).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models_dir", type=str, default=str(Path(__file__).resolve().parent / "models"))
    parser.add_argument("--results_dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = data_dir / "class_mapping.json"
    class_mapping, class_names = load_class_mapping(mapping_path)
    print(f"Loaded class mapping: {class_mapping}")
    print(f"Training on available dataset: {data_dir}")

    train_gen, val_gen, test_gen = get_data_generators(data_dir, args.batch_size, class_names=class_names)
    print(f"Classes: {train_gen.class_indices}")
    print(f"Train samples: {train_gen.samples} | Validation samples: {val_gen.samples} | Test samples: {test_gen.samples}")

    class_weight_dict = compute_class_weights(train_gen) if args.use_class_weights else None
    if class_weight_dict:
        print(f"Using class weights: {class_weight_dict}")

    model = build_cnn(input_shape=(224, 224, 3), num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = models_dir / f"best_model_{timestamp}.keras"
    final_model_path = models_dir / f"final_model_{timestamp}.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(str(best_model_path), monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_history(history, results_dir)
    plot_history(history, results_dir)

    test_metrics = model.evaluate(test_gen, verbose=1)
    metrics_dict = dict(zip(model.metrics_names, test_metrics))
    print(f"Test metrics: {metrics_dict}")
    evaluate_on_test(model, test_gen, results_dir)

    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")
    print(f"Saved best checkpoint to: {best_model_path}")

    best_model_path = best_model_path if best_model_path.exists() else final_model_path
    reloaded_model, test_predictions = verify_model(best_model_path, class_names, data_dir)

    metadata = {
        "model_architecture": "custom CNN",
        "num_classes": len(class_names),
        "class_mapping": class_mapping,
        "input_image_size": [224, 224, 3],
        "image_format": "RGB uint8 0-255",
        "normalization": "rescale=1/255 applied once in the data generator",
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": len(history.history["loss"]),
        "best_epoch": int(np.argmax(history.history["val_accuracy"])) + 1,
        "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "train_images": train_gen.samples,
        "validation_images": val_gen.samples,
        "test_images": test_gen.samples,
        "model_path": str(best_model_path),
        "note": "Training completed on the available 68-image sample/gallery dataset and not the full HAM10000 dataset.",
        "test_predictions": test_predictions,
    }
    (Path(__file__).resolve().parent / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run_summary = {
        "timestamp": timestamp,
        "data_dir": str(data_dir),
        "epochs_ran": len(history.history["loss"]),
        "best_epoch": int(np.argmax(history.history["val_accuracy"])) + 1,
        "test_metrics": metrics_dict,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "training_metrics": {"train_accuracy": history.history["accuracy"][-1], "val_accuracy": history.history["val_accuracy"][-1], "train_loss": history.history["loss"][-1], "val_loss": history.history["val_loss"][-1]},
    }
    (results_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("\n=== Team 3 verification complete ===")
    print(f"Best model loaded successfully: {best_model_path.exists()}")
    print(f"All {len(test_predictions)} test images generated predictions.")
    print("This model was trained and verified using the available 68-image sample/gallery dataset only.")


if __name__ == "__main__":
    main()
