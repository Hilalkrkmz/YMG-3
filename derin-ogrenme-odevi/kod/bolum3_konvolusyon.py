"""Bolum 3: Konvolusyon katmaninin kodlanmasi (hocanin verdigi kod + gorsellestirme)."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # gereksiz oneDNN mesajini kapatir
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # TensorFlow uyari yazilarini azaltir

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")          # pencere acmadan dosyaya kaydet
import matplotlib.pyplot as plt

CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cikti")
os.makedirs(CIKTI, exist_ok=True)

# Rastgelelik kontrolu
tf.keras.utils.set_random_seed(42)

# 3x3 boyutunda 16 filtreli konvolusyon katmani
conv = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
)

# Ornek giris goruntusu (1 adet, 128x128, 1 kanal)
dummy_input = tf.random.normal((1, 128, 128, 1))

# Konvolusyon islemi
output = conv(dummy_input)

# Agirliklari al
weights, biases = conv.get_weights()

print("Giris boyutu:", dummy_input.shape)
print("Cikis boyutu:", output.shape)
print("Filtre agirliklarinin boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)
print("Toplam parametre:", weights.size + biases.size)

print("Ilk filtre:")
print(weights[:, :, 0, 0])

# --- Ek: he_normal kontrolu (teorik: varyans = 2/fan_in, fan_in = 3*3*1 = 9) ---
print("He Normal teorik varyans (2/9):", round(2 / 9, 4))
print("Olculen varyans:", round(float(weights.var()), 4))
print("Olculen ortalama:", round(float(weights.mean()), 4))

# --- Ek: 16 filtrenin hepsini gorsellestir ---
fig, eksenler = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(eksenler.flat):
    ax.imshow(weights[:, :, 0, i], cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_title(f"Filtre {i}", fontsize=9)
    ax.axis("off")
plt.suptitle("16 filtrenin baslangic agirliklari (3x3, he_normal)")
plt.tight_layout()
yol = os.path.join(CIKTI, "filtreler.png")
plt.savefig(yol, dpi=150)
plt.close()

# Agirliklarin dagilimi (histogram)
plt.figure(figsize=(6, 4))
plt.hist(weights.flatten(), bins=20, edgecolor="black")
plt.title("144 agirligin dagilimi (he_normal)")
plt.xlabel("Agirlik degeri")
plt.ylabel("Adet")
plt.tight_layout()
plt.savefig(os.path.join(CIKTI, "agirlik_histogram.png"), dpi=150)
plt.close()

print("Kaydedildi:", os.path.abspath(yol))
