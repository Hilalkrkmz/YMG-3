"""Bolum 8: Ilk konvolusyonda 8 filtre ile 16 filtre deneyi."""
import os
import time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cikti")
os.makedirs(CIKTI, exist_ok=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
x_test = x_test.astype("float32")[..., np.newaxis] / 255.0


def model_kur(ilk_filtre):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(ilk_filtre, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


sonuclar = {}
gecmisler = {}
for f in (8, 16):
    print(f"\n===== Deney: ilk konvolusyonda {f} filtre =====")
    tf.keras.utils.set_random_seed(42)      # iki deney de ayni kosulda baslasin
    model = model_kur(f)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    baslangic = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                        validation_split=0.1, verbose=2)
    sure = time.time() - baslangic
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    sonuclar[f] = {
        "parametre": model.count_params(),
        "egitim_acc": history.history["accuracy"][-1],
        "val_acc": history.history["val_accuracy"][-1],
        "test_acc": test_acc,
        "sure": sure,
    }
    gecmisler[f] = history.history

print("\n" + "=" * 50)
print(f"{'Olcut':<20}{'Deney 1 (8)':>15}{'Deney 2 (16)':>15}")
satirlar = [
    ("Filtre sayisi", "8", "16"),
    ("Toplam parametre", sonuclar[8]["parametre"], sonuclar[16]["parametre"]),
    ("Egitim dogrulugu", f"{sonuclar[8]['egitim_acc']:.4f}", f"{sonuclar[16]['egitim_acc']:.4f}"),
    ("Dogrulama dogr.", f"{sonuclar[8]['val_acc']:.4f}", f"{sonuclar[16]['val_acc']:.4f}"),
    ("Test dogrulugu", f"{sonuclar[8]['test_acc']:.4f}", f"{sonuclar[16]['test_acc']:.4f}"),
    ("Egitim suresi (sn)", f"{sonuclar[8]['sure']:.1f}", f"{sonuclar[16]['sure']:.1f}"),
]
for ad, a, b in satirlar:
    print(f"{ad:<20}{str(a):>15}{str(b):>15}")

# Tabloyu dosyaya yaz (rapora kopyalamak icin)
with open(os.path.join(CIKTI, "deney_sonuclari.md"), "w", encoding="utf-8") as d:
    d.write("| Olcut | Deney 1 (8 filtre) | Deney 2 (16 filtre) |\n|---|---|---|\n")
    for ad, a, b in satirlar:
        d.write(f"| {ad} | {a} | {b} |\n")

# Karsilastirma grafigi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for f in (8, 16):
    plt.plot(gecmisler[f]["val_accuracy"], label=f"{f} filtre (val)")
plt.title("Dogrulama dogrulugu")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
for f in (8, 16):
    plt.plot(gecmisler[f]["val_loss"], label=f"{f} filtre (val)")
plt.title("Dogrulama loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CIKTI, "deney_karsilastirma.png"), dpi=150)
plt.close()
print("\nKaydedildi: cikti/deney_sonuclari.md ve cikti/deney_karsilastirma.png")
