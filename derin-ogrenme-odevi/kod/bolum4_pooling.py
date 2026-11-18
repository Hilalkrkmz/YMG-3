"""Bolum 4: Max Pooling isleminin incelenmesi."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# Ornek giris matrisi
input_data = tf.constant([
    [[1., 3., 2., 4.],
     [5., 6., 1., 2.],
     [7., 2., 8., 3.],
     [4., 1., 5., 9.]]
])

# Boyutlari duzenle: Batch, Height, Width, Channels
input_data = tf.reshape(input_data, (1, 4, 4, 1))

pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
output = pool(input_data)

print("Giris:")
print(input_data.numpy()[0, :, :, 0])
print("Pooling sonucu:")
print(output.numpy()[0, :, :, 0])
print("Giris boyutu:", input_data.shape)
print("Cikis boyutu:", output.shape)

# --- Ek: elle hesapladigimiz sonuc ile karsilastirma ---
# Sol ust: max(1,3,5,6)=6   Sag ust: max(2,4,1,2)=4
# Sol alt: max(7,2,4,1)=7   Sag alt: max(8,3,5,9)=9
elle = [[6., 4.], [7., 9.]]
print("Elle hesap:", elle)
print("Eslesiyor mu?", output.numpy()[0, :, :, 0].tolist() == elle)
