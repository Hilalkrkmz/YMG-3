"""Bolum 6-7: Fashion-MNIST ile basit CNN (ilk konvolusyonda 8 filtre)."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cikti")
os.makedirs(CIKTI, exist_ok=True)
tf.keras.utils.set_random_seed(42)

# A. Veri setinin yuklenmesi
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalizasyon (0-255 -> 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Kanal boyutu ekleme: (60000, 28, 28) -> (60000, 28, 28, 1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print("Egitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)

# B. CNN mimarisi
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.summary()

# C. Derleme ve egitim
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# D. Egitim sonuclarinin gorsellestirilmesi
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(CIKTI, "egitim_grafikleri.png"), dpi=150)
plt.close()
print("Grafik kaydedildi: cikti/egitim_grafikleri.png")

# --- Ek: Bolum 7 sorulari icin ara cikti boyutlari ---
print()
print("Katman katman cikti boyutlari:")
for katman in model.layers:
    print(f"  {katman.name:<20} {katman.output.shape}")
