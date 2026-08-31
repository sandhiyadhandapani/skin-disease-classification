"""
Team 3 - Deep Learning Model
model.py

Defines the CNN architecture used to classify the 7 HAM10000 skin lesion
classes from 224x224x3 images produced by Team 2's processing pipeline.

Team 2 already:
  - resized images to 224x224
  - converted BGR -> RGB
  - saved images as uint8 (0-255)

Team 3 must NOT re-normalize the saved files. Rescaling (1/255) is applied
only inside the data loader (see train_cnn.py), not here.
"""

from tensorflow.keras import layers, models, regularizers

# 7 HAM10000 classes (must match Team 2's folder names / class order)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)


def build_cnn(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES, l2_reg=1e-4):
    """
    Builds a compact CNN suitable for training from scratch on a small
    (e.g. 68-image) sample as well as the full HAM10000 dataset.

    Architecture: 4 conv blocks (Conv -> BatchNorm -> ReLU -> MaxPool),
    followed by GlobalAveragePooling and two dense layers with dropout.
    GlobalAveragePooling is used instead of Flatten to keep the parameter
    count low, which helps a lot when training on a tiny sample set.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    x = inputs
    filters = [32, 64, 128, 256]
    for i, f in enumerate(filters):
        x = layers.Conv2D(
            f, (3, 3), padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"conv_block{i + 1}_conv"
        )(x)
        x = layers.BatchNormalization(name=f"conv_block{i + 1}_bn")(x)
        x = layers.Activation("relu", name=f"conv_block{i + 1}_relu")(x)
        x = layers.MaxPooling2D((2, 2), name=f"conv_block{i + 1}_pool")(x)

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="skin_disease_cnn")
    return model


def build_transfer_model(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES,
                          base_trainable=False):
    """
    Optional stronger alternative: MobileNetV2 backbone pretrained on
    ImageNet, with a new classification head for the 7 classes.
    Use this instead of build_cnn() once the full dataset is available,
    since transfer learning generally beats a from-scratch CNN on
    medical image datasets of this size.
    """
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = base_trainable

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="skin_disease_mobilenetv2")
    return model


if __name__ == "__main__":
    # Quick sanity check: run `python model.py` to print the architecture.
    m = build_cnn()
    m.summary()
