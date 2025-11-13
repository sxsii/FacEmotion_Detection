# model_training.py
"""
Train a simple CNN on FER-like data.
This script supports:
 - fer2013.csv (original CSV format), OR
 - a directory layout: train/<label>/*.jpg and test/<label>/*.jpg

It also writes `class_names.json` to the project root so the web app can use
the exact mapping between numeric label indices and class names.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Settings (tweak for speed / memory)
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 6            # default small for smoke-test; increase for full training
MODEL_PATH = "face_emotionModel.h5"
CSV_PATH = "fer2013.csv"
CLASS_NAMES_JSON = "class_names.json"
AUTOTUNE = tf.data.AUTOTUNE

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=7):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_from_csv(csv_path):
    print("Loading dataset from CSV:", csv_path)
    data = pd.read_csv(csv_path)
    pixels = data['pixels'].tolist()
    emotions = data['emotion'].values
    X = []
    y = []
    for pix, emo in zip(pixels, emotions):
        arr = np.fromstring(pix, sep=' ', dtype=np.uint8)
        if arr.size != IMG_SIZE * IMG_SIZE:
            continue
        X.append(arr.reshape((IMG_SIZE, IMG_SIZE, 1)))
        y.append(emo)
    X = np.array(X, dtype='float32') / 255.0
    y = to_categorical(np.array(y, dtype=np.int32))
    # do a train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=np.argmax(y, axis=1))
    print("Shapes (train/val):", X_train.shape, X_val.shape)
    # for CSV mode we assume the classic FER label ordering 0..6 -> canonical names:
    canonical = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return (X_train, y_train), (X_val, y_val), canonical

def load_from_directories(train_dir='train', val_dir='test'):
    print(f"Loading dataset from directories. train='{train_dir}', val='{val_dir}'")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory '{train_dir}' not found.")
    if not os.path.isdir(val_dir):
        print(f"Validation directory '{val_dir}' not found; using 15% split from train.")
        val_dir = None

    # load training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    class_names = train_ds.class_names
    print("Detected class names (folder order -> label index):", class_names)
    num_classes = len(class_names)

    val_ds = None
    if val_dir:
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            labels='inferred',
            label_mode='int',
            image_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            shuffle=False
        )

    # normalization and one-hot encoding
    normalization = tf.keras.layers.Rescaling(1./255)

    def prep(ds, with_labels=True):
        ds = ds.map(lambda x, y: (normalization(x), tf.one_hot(y, depth=num_classes)), num_parallel_calls=AUTOTUNE)
        ds = ds.cache().prefetch(AUTOTUNE)
        return ds

    train_ds = prep(train_ds)
    if val_ds is not None:
        val_ds = prep(val_ds)
    else:
        # create val split from train_ds (rare case)
        raise RuntimeError("Validation directory missing - please provide 'test' directory or use CSV mode.")

    return train_ds, val_ds, class_names

def save_class_names(class_names):
    with open(CLASS_NAMES_JSON, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names to {CLASS_NAMES_JSON}")

def main():
    class_names = None

    # choose loading method
    if os.path.exists(CSV_PATH):
        (X_train, y_train), (X_val, y_val), class_names = load_from_csv(CSV_PATH)
        num_classes = y_train.shape[1]
        model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=num_classes)
        checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
        early = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[checkpoint, early], verbose=1
        )
    else:
        # directory mode
        if os.path.isdir('train') and os.path.isdir('test'):
            train_ds, val_ds, class_names = load_from_directories('train', 'test')
            num_classes = len(class_names)
            model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=num_classes)
            checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
            early = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=[checkpoint, early],
                verbose=1
            )
        else:
            raise FileNotFoundError("No fer2013.csv and no train/test directories found. Place either CSV or train/test folders in the project root.")

    # save model and class names
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    if class_names is None:
        class_names = ['class_' + str(i) for i in range(model.output_shape[-1])]

    save_class_names(class_names)
    print("Class names (folder -> index):", class_names)

if __name__ == "__main__":
    main()
