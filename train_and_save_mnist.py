"""
MNIST (el yazısı rakam) ile CNN model eğitme + kaydetme

Bu script:
1) MNIST verisini yükler
2) Ön işleme yapar (normalize + kanal ekleme)
3) CNN modeli kurar
4) Eğitir (EarlyStopping + ModelCheckpoint ile)
5) En iyi modeli .keras formatında kaydeder
6) Test setinde sonucu yazdırır
"""

import os
import numpy as np
import tensorflow as tf

# ----------------------------
# 0) Tekrarlanabilirlik (opsiyonel ama faydalı)
# ----------------------------
# Eğitim her çalıştırmada ufak farklılık gösterebilir.
# Bu seed'ler "tam aynı" garantisi vermez (GPU/ops), ama dalgalanmayı azaltır.
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Eğer CPU'da deterministik davranış istiyorsan bazen işe yarar.
# (Her sistemde %100 garanti değil, ama tutarlılık artırır.)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# ----------------------------
# 1) Veri setini yükle (MNIST)
# ----------------------------
# mnist.load_data() -> ((x_train, y_train), (x_test, y_test))
# x_* : görüntüler (uint8, 0-255) shape: (N, 28, 28)
# y_* : etiketler (0-9) shape: (N,)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ----------------------------
# 2) Ön işleme (Preprocessing)
# ----------------------------

# 2.1) Normalize et: 0-255 -> 0-1 aralığına çek
# Neden? Eğitim daha stabil/kolay olur.
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# 2.2) CNN için kanal boyutu ekle:
# Şu an shape (N, 28, 28). Conv2D genelde (N, 28, 28, 1) bekler (channels_last).
x_train = x_train[..., np.newaxis]  # np.newaxis == None ile aynı iş
x_test  = x_test[..., np.newaxis]

# Kontrol amaçlı (istersen aç)
# print("x_train shape:", x_train.shape)  # (60000, 28, 28, 1)
# print("y_train shape:", y_train.shape)  # (60000,)

# ----------------------------
# 3) Modeli kur (CNN)
# ----------------------------
# Sequential: katmanları sırayla dizdiğimiz model türü
model = tf.keras.Sequential([
    # Conv2D parametreleri:
    # - filters=32: bu katmanda 32 adet filtre (özellik haritası) öğrenecek
    # - kernel_size=(3,3): filtre boyutu 3x3
    # - activation="relu": negatifleri 0'a kırpar, öğrenmeyi hızlandırır
    # - input_shape=(28,28,1): ilk katman için giriş şekli
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    # MaxPooling2D:
    # - pool_size=(2,2): her 2x2 bölgede max alır, boyutu küçültür
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Daha derin özellikler için ikinci conv bloğu
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten: (h, w, c) -> tek boyutlu vektör
    tf.keras.layers.Flatten(),

    # Dense (tam bağlı) katman:
    # - units=128: 128 nöron
    # - activation="relu"
    tf.keras.layers.Dense(128, activation="relu"),

    # Çıkış katmanı:
    # - units=10: 0-9 => 10 sınıf
    # - softmax: olasılık dağılımı üretir (toplam 1)
    tf.keras.layers.Dense(10, activation="softmax")
])

# İstersen model özetini gör
model.summary()

# ----------------------------
# 4) Compile (Model nasıl öğrenecek?)
# ----------------------------
model.compile(
    # optimizer="adam": pratikte çok iyi bir default optimizer
    optimizer="adam",

    # loss="sparse_categorical_crossentropy":
    # Etiketler one-hot değil, direkt 0-9 olduğu için "sparse" doğru.
    loss="sparse_categorical_crossentropy",

    # metrics=["accuracy"]: eğitim boyunca doğruluğu raporlar
    metrics=["accuracy"]
)

# ----------------------------
# 5) Callback'ler: Eğitimi daha akıllı yapmak
# ----------------------------

# 5.1) EarlyStopping:
# validation loss iyileşmeyi bırakınca eğitimi durdurur.
# - monitor="val_loss": doğrulamayı (validation) takip et
# - patience=3: 3 epoch üst üste iyileşme yoksa dur
# - restore_best_weights=True: en iyi noktadaki ağırlıkları geri yükle
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 5.2) ModelCheckpoint:
# En iyi modeli diske kaydeder.
# - filepath: kaydedilecek dosya adı
# - monitor="val_loss": validation loss en iyiyken kaydet
# - save_best_only=True: sadece "en iyi" olunca üzerine yaz
# Not: .keras formatı Keras'ın modern, tavsiye edilen kaydetme formatı.
best_model_path = "mnist_cnn_best.keras"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_loss",
    save_best_only=True
)

# ----------------------------
# 6) Eğitim (fit)
# ----------------------------
history = model.fit(
    # Eğitim verisi
    x_train, y_train,

    # epochs: maksimum epoch sayısı (EarlyStopping daha erken durdurabilir)
    epochs=20,

    # batch_size: bir seferde kaç örnek işlensin (default 32)
    # GPU varsa 64/128 genelde daha hızlı olabilir.
    batch_size=64,

    # validation_split: eğitim verisinin bir kısmını doğrulama için ayırır
    # Alternatif: validation_data=(x_test, y_test) diyebilirsin.
    validation_split=0.1,

    # Callback'ler
    callbacks=[early_stopping, checkpoint],

    # verbose=1: eğitim ilerlemesini ekrana basar
    verbose=1
)

# ----------------------------
# 7) En iyi modeli yükle (garanti olsun diye)
# ----------------------------
# Eğitim sonunda modelin ağırlıkları zaten restore_best_weights ile geri dönebilir,
# ama checkpoint dosyasından yüklemek "kayıt gerçekten var mı" kontrolü gibi düşün.
best_model = tf.keras.models.load_model(best_model_path)

# ----------------------------
# 8) Test setinde değerlendir
# ----------------------------
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ----------------------------
# 9) İstersen ayrıca final modeli farklı isimle de kaydet
# ----------------------------
# Bu, "best model" dosyasını kopya gibi düşünebilirsin.
final_model_path = "mnist_cnn_final.keras"
best_model.save(final_model_path)
print(f"\nEn iyi model kaydedildi: {best_model_path}")
print(f"Final model kaydedildi:  {final_model_path}")

# ----------------------------
# 10) Mini demo: testten bir örnek tahmin et
# ----------------------------
# Tek örnek: batch boyutu için [i:i+1] kullanıyoruz.
i = 0
probs = best_model.predict(x_test[i:i+1], verbose=0)  # shape: (1,10)
pred_digit = int(np.argmax(probs, axis=1)[0])         # en yüksek olasılığın index'i
true_digit = int(y_test[i])

print(f"\nÖrnek #{i} -> Tahmin: {pred_digit} | Gerçek: {true_digit}")
print("Olasılıklar:", np.round(probs[0], 3))
