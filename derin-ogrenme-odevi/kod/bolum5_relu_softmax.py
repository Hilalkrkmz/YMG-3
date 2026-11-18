"""Bolum 5: ReLU ve Softmax fonksiyonlarinin incelenmesi."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# ---- A. ReLU ----
x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
relu = tf.keras.layers.ReLU()
y = relu(x)
print("Giris:", x.numpy())
print("ReLU cikisi:", y.numpy())

# ---- B. Softmax ----
logits = tf.constant([[2.0, 1.0, 0.1]])
softmax = tf.keras.layers.Softmax()
probabilities = softmax(logits)
print("Softmax sonuclari:")
print(probabilities.numpy())
print("Olasiliklarin toplami:")
print(tf.reduce_sum(probabilities).numpy())

# ---- Ek: elle kontrol ----
e = np.exp([2.0, 1.0, 0.1])
print("Elle hesap (e^zi / toplam):", np.round(e / e.sum(), 4))
print("En yuksek olasilikli sinif (argmax):", int(np.argmax(probabilities.numpy())))
