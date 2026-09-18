import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (865, 285)
BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 0.00005

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "vidalar"
LABELS_FILE = DATA_DIR / "labels.csv"
MODEL_FILE = Path("cnn_sifirdan.keras")


def preprocess_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    son = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    med_son = cv2.medianBlur(son, 3)
    arka_plan = cv2.medianBlur(son, 37)
    maske = cv2.addWeighted(med_son, 1, arka_plan, -1, 255)
    return cv2.bitwise_and(maske, med_son)


def load_dataset():
    df = pd.read_csv(LABELS_FILE, sep=",")

    files = sorted(
        file.name
        for file in IMAGE_DIR.iterdir()
        if file.is_file()
    )

    img_list = []
    for filename in files:
        image = cv2.imread(str(IMAGE_DIR / filename))
        if image is None:
            raise ValueError(f"Görüntü okunamadı: {filename}")
        img_list.append(preprocess_image(image))

    x = np.array(img_list)
    y = df["durum"].to_numpy()

    csv_ids = df["id"].tolist()
    if files != csv_ids:
        raise ValueError("Görüntü dosyaları ile CSV 'id' değerleri eşleşmiyor.")

    return x, y


def build_model():
    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),

        Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal", activation="swish"),
        BatchNormalization(),

        Conv2D(64, 3, activation="swish"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(32, 3, activation="swish"),
        BatchNormalization(),
        Conv2D(32, 3, activation="swish"),
        BatchNormalization(),
        Conv2D(32, 3, activation="swish"),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="swish"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1),
        Activation("sigmoid"),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def train():
    x, y = load_dataset()

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=42,
        shuffle=True,
    )

    datagen = ImageDataGenerator(horizontal_flip=False, vertical_flip=False)
    data_generator = datagen.flow(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        seed=2020,
    )

    model = build_model()
    model.summary()

    lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=3,
        verbose=1,
        mode="auto",
        factor=0.25,
        min_lr=0.000001,
    )

    history = model.fit(
        data_generator,
        steps_per_epoch=1000,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[lr],
    )

    model.save(MODEL_FILE)
    return model, history


def predict_video(model, video_path, output_path="sonucSifirdanModel.avi"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video açılamadı: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess_image(frame)
        processed = cv2.resize(processed, IMAGE_SIZE)
        processed = processed.astype(np.float32)
        processed = np.expand_dims(processed, axis=0)

        prediction = model.predict(processed, verbose=0)[0][0]
        result_text = "Kusurlu" if prediction < 0.5 else "Kusursuz"

        cv2.putText(
            frame,
            result_text,
            (frame_width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    train()
