"""
Team 3 - Deep Learning Model
train_cnn.py

Loads Team 2's processed TEAM2_OUTPUT (train/validation/test folders),
trains the CNN defined in model.py, evaluates it, and saves the final
.keras model for Team 4.

Usage (from inside team3-deep-learning/):
    python train_cnn.py --data_dir ../TEAM2_OUTPUT --epochs 30

Assumptions (per Team 2's handover doc — do not change these):
  - Images inside TEAM2_OUTPUT/train, /validation, /test are already
    224x224, RGB, saved as uint8 (0-255). Team 3 rescales by 1/255 only
    here, in the data loader — never re-saves normalized images.
  - Folder layout is class-per-subfolder (Keras "flow_from_directory"
    style):
        TEAM2_OUTPUT/train/akiec/*.jpg, TEAM2_OUTPUT/train/bcc/*.jpg, ...
  - Team 2 already applies augmentation (rotation/flip/zoom) to the
    train images they generate, and validation/test get NO augmentation.
    Because of that, Team 3's ImageDataGenerator below only rescales —
    it does NOT add a second layer of augmentation on top of Team 2's.
    If Team 2's output is *not* pre-augmented and you want Team 3 to
    handle augmentation instead, flip AUGMENT_IN_TEAM3 to True below.
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # safe for headless / VS Code terminal runs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import CLASS_NAMES, IMG_SIZE, NUM_CLASSES, build_cnn

# Set to True only if Team 2's saved images are NOT already augmented
# and you want Team 3's generator to augment the training set instead.
AUGMENT_IN_TEAM3 = False


def get_data_generators(data_dir, batch_size):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    for d in (train_dir, val_dir, test_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Expected Team 2 output folder not found: {d}\n"
                "Make sure TEAM2_OUTPUT/train, /validation and /test exist "
                "(see --data_dir)."
            )

    if AUGMENT_IN_TEAM3:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.15,
        )
    else:
        # Team 2 already augmented the train split; only rescale here.
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Validation and test must NEVER be augmented.
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42,
    )
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def compute_class_weights(train_gen):
    """Handles class imbalance, which HAM10000 has a lot of (nv dominates)."""
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


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
    path = os.path.join(out_dir, "training_history.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved training curves to {path}")


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

    with open(os.path.join(out_dir, "test_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    # Simple confusion matrix plot
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
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close(fig)

    return report


def main():
    parser = argparse.ArgumentParser(description="Train Team 3 CNN on Team 2 processed data")
    parser.add_argument("--data_dir", type=str, default="../TEAM2_OUTPUT",
                         help="Path to TEAM2_OUTPUT folder (contains train/validation/test)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Use a small batch size (e.g. 4-8) when training on the 68-image sample")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading data from: {args.data_dir}")
    train_gen, val_gen, test_gen = get_data_generators(args.data_dir, args.batch_size)

    print(f"Classes found: {train_gen.class_indices}")
    print(f"Train samples: {train_gen.samples} | "
          f"Val samples: {val_gen.samples} | Test samples: {test_gen.samples}")

    class_weights = compute_class_weights(train_gen) if args.use_class_weights else None
    if class_weights:
        print(f"Using class weights: {class_weights}")

    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(args.models_dir, f"best_model_{timestamp}.keras")
    final_model_path = os.path.join(args.models_dir, f"skin_disease_cnn_{timestamp}.keras")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    plot_history(history, args.results_dir)

    print("\nEvaluating on the held-out test set...")
    test_metrics = model.evaluate(test_gen, verbose=1)
    metrics_dict = dict(zip(model.metrics_names, test_metrics))
    print(f"Test metrics: {metrics_dict}")

    evaluate_on_test(model, test_gen, args.results_dir)

    model.save(final_model_path)
    print(f"\nSaved final model to: {final_model_path}")
    print(f"Saved best checkpoint to: {best_model_path}")
    print("Hand these .keras files to Team 4 for evaluation/prediction.")

    with open(os.path.join(args.results_dir, "run_summary.json"), "w") as f:
        json.dump({
            "timestamp": timestamp,
            "data_dir": args.data_dir,
            "epochs_ran": len(history.history["loss"]),
            "test_metrics": metrics_dict,
            "final_model_path": final_model_path,
            "best_model_path": best_model_path,
        }, f, indent=2)


if __name__ == "__main__":
    main()
