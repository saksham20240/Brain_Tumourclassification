"""
Brain Tumor Classification pipeline - plain Python script (no notebook).

Expected folder layout (as you already have it):

    Brain_Tumour/
        dataset/
            Training/{glioma,meningioma,notumor,pituitary}/*.jpg
            Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
        brain_tumor_pipeline.py   <- this file

Run it from inside Brain_Tumour/ (conda env activated):

    conda activate brain-tumor-gpu
    python brain_tumor_pipeline.py --stage all

Stages (run with --stage <name>, default "all"):
    preprocess  - crop + resize raw images into Crop-Brain-MRI/ and Test-Data/
    train       - build EfficientNetB1 model and fit it, saving model.keras
    evaluate    - confusion matrix + classification report on Test-Data
    predict     - batched prediction over Test-Data + a sample grid image
    gradcam     - Grad-CAM visualization for one example image
    all         - run every stage in order

Preprocessing is skipped automatically if Crop-Brain-MRI/Test-Data already
have images in them, so you can safely re-run the script (e.g. `--stage
train`) without re-cropping every time. Use --force-preprocess to redo it.
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; all plots are saved to files
import matplotlib.pyplot as plt
import imutils
from tqdm import tqdm


# ---------------------------------------------------------------------------
# GPU setup - must run before any TF op touches the GPU.
# ---------------------------------------------------------------------------
def configure_gpu():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s): {[g.name for g in gpus]} - memory growth enabled.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU visible to TensorFlow - training will fall back to CPU and be very slow.")

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", mixed_precision.global_policy())
    return tf


# ---------------------------------------------------------------------------
# Cropping (same logic as the original notebook)
# ---------------------------------------------------------------------------
def crop_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.erode(img_thresh, None, iterations=2)
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)

    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()])[0]
    extRight = tuple(c[c[:, :, 0].argmax()])[0]
    extTop = tuple(c[c[:, :, 1].argmin()])[0]
    extBottom = tuple(c[c[:, :, 1].argmax()])[0]

    return image[extTop[1]:extBottom[1], extLeft[0]:extRight[0]]


CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def _crop_and_save_folder(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    j = 0
    for fname in tqdm(os.listdir(src_dir), desc=f"{src_dir.name}/{dst_dir.parent.name}"):
        img = cv2.imread(str(src_dir / fname))
        if img is None:
            continue
        cropped = crop_image(img)
        if cropped is None or cropped.size == 0:
            continue
        cropped = cv2.resize(cropped, (240, 240))
        cv2.imwrite(str(dst_dir / f"{j}.jpg"), cropped)
        j += 1


def preprocess(base_dir: Path, dataset_dir: Path, force: bool = False):
    train_src = dataset_dir / "Training"
    test_src = dataset_dir / "Testing"
    crop_dst = base_dir / "Crop-Brain-MRI"
    test_dst = base_dir / "Test-Data"

    already_done = crop_dst.exists() and any(
        (crop_dst / c).exists() and any((crop_dst / c).iterdir()) for c in CLASSES
    )
    if already_done and not force:
        print(f"{crop_dst} already has cropped images - skipping preprocessing "
              f"(use --force-preprocess to redo it).")
        return

    for cls in CLASSES:
        _crop_and_save_folder(train_src / cls, crop_dst / cls)
    for cls in CLASSES:
        _crop_and_save_folder(test_src / cls, test_dst / cls)

    print("Preprocessing done:")
    for cls in CLASSES:
        print(f"  train/{cls}: {len(os.listdir(crop_dst / cls))} images, "
              f"test/{cls}: {len(os.listdir(test_dst / cls))} images")


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def build_generators(base_dir: Path, batch_size: int):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=10,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_data = datagen.flow_from_directory(
        str(base_dir / "Crop-Brain-MRI"),
        target_size=(240, 240),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )
    valid_data = datagen.flow_from_directory(
        str(base_dir / "Crop-Brain-MRI"),
        target_size=(240, 240),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    test_datagen = ImageDataGenerator()
    test_data = test_datagen.flow_from_directory(
        str(base_dir / "Test-Data"),
        target_size=(240, 240),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_data, valid_data, test_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model():
    from tensorflow.keras.applications import EfficientNetB1
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D

    effnet = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(240, 240, 3))
    x = effnet.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    # fp32 output layer keeps softmax/loss numerically stable under mixed precision
    x = Dense(4, activation="softmax", dtype="float32")(x)
    model = Model(inputs=effnet.input, outputs=x)
    return model


def train(base_dir: Path, batch_size: int, epochs: int):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    train_data, valid_data, _ = build_generators(base_dir, batch_size)

    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.0001),
                   loss="categorical_crossentropy", metrics=["accuracy"])

    model_path = base_dir / "model.keras"
    checkpoint = ModelCheckpoint(str(model_path), monitor="val_accuracy",
                                  save_best_only=True, mode="auto", verbose=1)
    earlystop = EarlyStopping(monitor="val_accuracy", patience=5, mode="auto", verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=2,
                                   min_delta=0.001, mode="auto", verbose=1)

    history = model.fit(
        train_data, epochs=epochs, validation_data=valid_data,
        verbose=1, callbacks=[checkpoint, earlystop, reduce_lr],
    )

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    ax[0].plot(epochs_range, history.history["accuracy"], "g-o", label="Training Accuracy")
    ax[0].plot(epochs_range, history.history["val_accuracy"], "y-o", label="Validation Accuracy")
    ax[0].set_title("Model Training & Validation Accuracy")
    ax[0].legend(loc="lower right")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs_range, history.history["loss"], "g-o", label="Training loss")
    ax[1].plot(epochs_range, history.history["val_loss"], "y-o", label="Validation loss")
    ax[1].set_title("Model Training & Validation loss")
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    out_path = base_dir / "training_history.png"
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")
    print(f"Best model saved to {model_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(base_dir: Path, batch_size: int):
    from tensorflow.keras.models import load_model
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

    _, _, test_data = build_generators(base_dir, batch_size)
    model = load_model(str(base_dir / "model.keras"))

    print("Test set evaluation:", model.evaluate(test_data))

    y_test = test_data.classes
    y_pred = np.argmax(model.predict(test_data), axis=1)

    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    cm_display.plot()
    out_path = base_dir / "confusion_matrix.png"
    plt.savefig(out_path)
    print(f"Saved confusion matrix to {out_path}")

    report = classification_report(y_test, y_pred, target_names=CLASSES)
    print(report)
    (base_dir / "classification_report.txt").write_text(report)


# ---------------------------------------------------------------------------
# Prediction on test images (batched)
# ---------------------------------------------------------------------------
def predict(base_dir: Path, batch_size: int):
    import PIL.Image
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from sklearn.metrics import accuracy_score

    model = load_model(str(base_dir / "model.keras"))
    test_dir = base_dir / "Test-Data"

    images, original, batch = [], [], []
    for cls in os.listdir(test_dir):
        for item in os.listdir(test_dir / cls):
            img = PIL.Image.open(test_dir / cls / item).convert("RGB").resize((240, 240))
            images.append(img)
            batch.append(img_to_array(img))
            original.append(cls)

    batch = np.array(batch, dtype=np.float32)

    # Predict in manual chunks. Passing the whole array to model.predict()
    # with batch_size=... still converts the entire array to one big GPU
    # tensor before slicing it internally, which OOMs on a 4GB card. Slicing
    # it ourselves means only `batch_size` images ever hit the GPU at once.
    preds_list = []
    for i in tqdm(range(0, len(batch), batch_size), desc="predict"):
        chunk = batch[i:i + batch_size]
        preds_list.append(model.predict(chunk, verbose=0))
    preds = np.concatenate(preds_list, axis=0)
    prediction = [CLASSES[p] for p in np.argmax(preds, axis=1)]

    score = accuracy_score(original, prediction)
    print(f"Overall accuracy on Test-Data: {score:.4f}")

    fig = plt.figure(figsize=(20, 20))
    for i in range(10):
        j = random.randint(0, len(images) - 1)
        fig.add_subplot(5, 2, i + 1)
        plt.xlabel(f"Prediction: {prediction[j]}   Original: {original[j]}")
        plt.imshow(images[j])
    fig.tight_layout()
    out_path = base_dir / "sample_predictions.png"
    fig.savefig(out_path)
    print(f"Saved sample predictions grid to {out_path}")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def viz_gradcam(model, image, tf, save_path, interpolant=0.5):
    from tensorflow.keras.models import Model as KModel

    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    original_img = image
    img = np.expand_dims(original_img, axis=0)

    with tf.GradientTape() as tape:
        gradient_model = KModel([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        prediction_idx = np.argmax(prediction)
        loss = prediction[:, prediction_idx]

    gradients = tape.gradient(loss, conv2d_out)
    # conv2d_out/gradients are float16 under the mixed_precision policy;
    # convert to plain float32 numpy up front so the accumulation below and
    # cv2.resize() get a dtype they both understand.
    output = conv2d_out[0].numpy().astype(np.float32)
    weights = tf.reduce_mean(gradients[0], axis=(0, 1)).numpy().astype(np.float32)

    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]

    activation_map = cv2.resize(activation_map, (original_img.shape[1], original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    activation_map = np.uint8(255 * activation_map)
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    original_img = np.uint8((original_img - original_img.min()) /
                             (original_img.max() - original_img.min() + 1e-8) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
    plt.axis("off")
    plt.savefig(save_path)
    print(f"Saved Grad-CAM visualization to {save_path}")


def gradcam(base_dir: Path, tf):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array

    model = load_model(str(base_dir / "model.keras"))
    example_path = next((base_dir / "Test-Data" / "glioma").iterdir())
    test_img = cv2.imread(str(example_path))
    viz_gradcam(model, img_to_array(test_img), tf, base_dir / "gradcam_example.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default=".", help="Path to the Brain_Tumour folder (default: current dir)")
    parser.add_argument("--dataset-dir", default=None,
                         help="Path to the dataset folder (default: <base-dir>/dataset)")
    parser.add_argument("--stage", default="all",
                         choices=["preprocess", "train", "evaluate", "predict", "gradcam", "all"])
    parser.add_argument("--batch-size", type=int, default=8,
                         help="Lower to 4 if you hit an OOM on a 4GB GPU (default: 8)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force-preprocess", action="store_true",
                         help="Re-crop/resize images even if Crop-Brain-MRI/Test-Data already exist")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else base_dir / "dataset"

    tf = configure_gpu()

    stages = [args.stage] if args.stage != "all" else ["preprocess", "train", "evaluate", "predict", "gradcam"]

    for stage in stages:
        print(f"\n=== Stage: {stage} ===")
        if stage == "preprocess":
            preprocess(base_dir, dataset_dir, force=args.force_preprocess)
        elif stage == "train":
            train(base_dir, args.batch_size, args.epochs)
        elif stage == "evaluate":
            evaluate(base_dir, args.batch_size)
        elif stage == "predict":
            predict(base_dir, args.batch_size)
        elif stage == "gradcam":
            gradcam(base_dir, tf)


if __name__ == "__main__":
    main()
