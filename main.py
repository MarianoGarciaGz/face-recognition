# main.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.layers import Input
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# Configuración de directorios
data_dir = "./"  # Ruta al dataset
categories = ["happy_person_face", "sad_person_face"]

# Preprocesamiento de imágenes
img_size = (128, 128)
batch_size = 32

# Generadores de datos
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.3)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
)


# Generador personalizado para repetición indefinida
def infinite_generator(generator):
    while True:
        for batch in generator:
            yield batch


train_generator = infinite_generator(train_generator)
validation_generator = infinite_generator(validation_generator)

# Construcción del Modelo
model = tf.keras.models.Sequential(
    [
        Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Compilación del Modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Calcular steps_per_epoch y validation_steps
steps_per_epoch = (
    train_generator.samples // batch_size
    if train_generator.samples // batch_size > 0
    else 1
)
validation_steps = (
    validation_generator.samples // batch_size
    if validation_generator.samples // batch_size > 0
    else 1
)

# Entrenamiento del Modelo
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=10,
    validation_data=validation_generator,
)

# Evaluación del Modelo
loss, accuracy = model.evaluate(validation_generator, steps=validation_steps)
print(f"Precisión del modelo: {accuracy*100:.2f}%")

# Guardar el modelo entrenado
model.save("face_classification_model.h5")


# Interfaz de Usuario Simple
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = load_img(file_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        result = "Happy" if prediction[0][0] > 0.5 else "Sad"
        result_label.config(text=f"Prediction: {result}")
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img


root = tk.Tk()
root.title("Image Classifier")

upload_button = tk.Button(root, text="Upload Image", command=classify_image)
upload_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

panel = tk.Label(root)
panel.pack()

root.mainloop()
